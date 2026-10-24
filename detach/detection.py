import cv2
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from pycocotools import mask
import networkx as nx
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk
from sunpy.coordinates import frames
from sunpy.sun.constants import au, equatorial_radius
from aiapy.calibrate import correct_degradation
from skimage.exposure import equalize_hist
from skimage.draw import polygon, polygon2mask
from sqlalchemy.orm import Session
from sqlalchemy import func
from mmdet.apis import inference_detector
from mmengine.config import ConfigDict

from .database import Instance

# r sun
r_sun = equatorial_radius.value/1e6  # unit:Mm

# ignore verbose sunpy radius-missing log and RFA warnings
import logging
import warnings
logging.getLogger('sunpy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", module='erfa')

def remove_overlap(origin_result, overlap_threshold=0.6):
    id_mask = np.argwhere(np.array(origin_result['scores'])>0).reshape(-1)
    prediction_result = {'labels':[], 'scores':[], 'bboxes':[], 'masks':[]}
    G = nx.Graph()
    G.add_nodes_from(id_mask)
    for i, j in combinations(id_mask, 2):
        buffer_i = mask.decode(origin_result['masks'][i])
        buffer_j = mask.decode(origin_result['masks'][j])
        overlap = np.sum(buffer_i*buffer_j)
        # iou = overlap/(np.sum(buffer_i)+np.sum(buffer_j)-overlap)
        ratio = overlap/np.min([np.sum(buffer_i), np.sum(buffer_j)])
        if ratio > overlap_threshold:
            G.add_edge(i, j)
    connected_components = list(nx.connected_components(G))
    for component in connected_components:
        component = np.array(list(component)) # change type to numpy array
        if len(component) == 1:
            origin_id = component[0]
            for key in origin_result.keys():
                prediction_result[key].append(origin_result[key][origin_id])
        else:
            label_list = np.array(origin_result['labels'])[component]
            score_list = np.array(origin_result['scores'])[component]
            final_label = label_list[np.argmax(score_list)]
            prediction_result['labels'].append(final_label)         # final label
            prediction_result['scores'].append(np.max(score_list))  # final score
            component = component[label_list == final_label]
            buffer = np.zeros(shape=origin_result['masks'][0]['size'], order='F', dtype=np.bool_)
            for id in component:
                buffer = buffer + mask.decode(origin_result['masks'][id])
            buffer[buffer>1] = 1
            prediction_result['masks'].append(mask.encode(buffer))          # final mask
            ymax, xmax = np.max(np.where(buffer>0), 1)
            ymin, xmin = np.min(np.where(buffer>0), 1)
            prediction_result['bboxes'].append([xmin, ymin, xmax, ymax])    # final bbox
    return prediction_result

def polar_correction(origin_result, aia_map, polar_num_threshold=1e4, ty_threshold=850):
    id_mask = np.argwhere((np.array(origin_result['labels'])==1)).reshape(-1)
    for id in id_mask:
        y_list, x_list = np.where(mask.decode(origin_result['masks'][id]) > 0)
        ty = aia_map.pixel_to_world(x_list*u.pix, y_list*u.pix).Ty.value
        polar_num = np.sum(np.abs(ty)>ty_threshold)
        if polar_num > polar_num_threshold:
            origin_result['labels'][id] = 0
    return origin_result

def cal_spherical_info(phi_list, theta_list):  # phi->longitude  theta->latitude
    phi_list = np.asarray(phi_list, dtype=float) * np.pi / 180  # 转换成弧度
    theta_list = np.asarray(theta_list, dtype=float) * np.pi / 180

    # --- unwrap phi to handle boundary crossing the HGS ±180° line ---
    # When the CH straddles ±180°, consecutive boundary points have
    # phi differences of ~±2π (e.g. +179° → -179°).  Unwrap so that
    # adjacent points differ by at most π, giving a continuous sequence.
    for i in range(1, len(phi_list)):
        diff = phi_list[i] - phi_list[i - 1]
        if diff > np.pi:
            phi_list[i] -= 2 * np.pi
        elif diff < -np.pi:
            phi_list[i] += 2 * np.pi

    shifted_phi = np.roll(phi_list, 1)
    shifted_theta = np.roll(theta_list, 1)
    delta_theta = theta_list - shifted_theta
    same_id = (delta_theta == 0)
    S0 = (phi_list[same_id] - shifted_phi[same_id])*np.sin(theta_list[same_id])
    theta0 = (phi_list[same_id] - shifted_phi[same_id])*(theta_list[same_id]*np.sin(theta_list[same_id])+np.cos(theta_list[same_id]))
    phi0 = 0.5*np.sin(theta_list[same_id])*(phi_list[same_id]**2-shifted_phi[same_id]**2)

    S1 = -(phi_list[~same_id] - shifted_phi[~same_id]) \
        *(np.cos(theta_list[~same_id])-np.cos(shifted_theta[~same_id])) \
            /(theta_list[~same_id]-shifted_theta[~same_id])
    theta1 = (phi_list[~same_id] - shifted_phi[~same_id]) \
        /(theta_list[~same_id]-shifted_theta[~same_id]) \
            *(2*np.sin(theta_list[~same_id])-theta_list[~same_id]*np.cos(theta_list[~same_id]) \
              -2*np.sin(shifted_theta[~same_id])+shifted_theta[~same_id]*np.cos(shifted_theta[~same_id]))
    A = (phi_list[~same_id]**2-shifted_phi[~same_id]**2)/2/(theta_list[~same_id]-shifted_theta[~same_id]) \
        -(theta_list[~same_id]+shifted_theta[~same_id])/2*((phi_list[~same_id]-shifted_phi[~same_id])/(theta_list[~same_id]-shifted_theta[~same_id]))**2
    B = ((phi_list[~same_id]-shifted_phi[~same_id])/(theta_list[~same_id]-shifted_theta[~same_id]))**2
    phi1 = -A*(np.cos(theta_list[~same_id])-np.cos(shifted_theta[~same_id])) \
        +B*(-theta_list[~same_id]*np.cos(theta_list[~same_id])+np.sin(theta_list[~same_id]) \
            + shifted_theta[~same_id]*np.cos(shifted_theta[~same_id]) - np.sin(shifted_theta[~same_id]))
    S = np.sum(S0)+np.sum(S1)
    theta = np.sum(theta0) + np.sum(theta1)
    phi = np.sum(phi0) + np.sum(phi1)
    if S != 0:
        theta = theta/S
        phi = phi/S
        theta = theta * 180/np.pi
        phi = phi * 180 / np.pi
        # wrap centroid phi back to [-180, 180]
        phi = (phi + 180.0) % 360.0 - 180.0
        return np.abs(S)/(4*np.pi), phi, theta
    else:
        return 0, None, None


class Detection(object):
    def __init__(self, fits_path, detection_model,
                 detection_threshold=0.3, frame_id=0,
                 recover_degrad=True, recover_limb=True, equ_hist=True, 
                 polar_correction=True, remove_overlap=True) -> None:
        self.solar_map = Map(fits_path)
        self.img_shape = self.solar_map.data.shape[:2]
        self.fits_path = fits_path
        self.detection_model = detection_model
        self.detection_threshold = detection_threshold
        self.frame_id = frame_id
        self.recover_degrad = recover_degrad
        self.recover_limb = recover_limb
        self.equ_hist = equ_hist
        self.polar_correction = polar_correction
        self.remove_overlap = remove_overlap
        self.rs = (equatorial_radius/(au)*u.rad).to(u.arcsec)  # 以arcsec为单位的太阳半径

    def init_detect_array(self):
        if type(self.recover_degrad) in [dict, ConfigDict]:
            # corr_table_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'aia_V10_20201119_190000_response_table.txt')
            self.solar_map = correct_degradation(self.solar_map, **(self.recover_degrad))
        elif self.recover_degrad:
            self.solar_map = correct_degradation(self.solar_map)
        self.detect_array = self.solar_map.data
        if self.recover_limb or type(self.recover_limb) is dict:
            self.remove_limb_brightness()
        if self.equ_hist or type(self.equ_hist) is dict:
            if type (self.equ_hist) is not dict:
                self.equ_hist = {}
            self.detect_array = equalize_hist(self.detect_array, **(self.equ_hist))*255  # data range [0-255]

    def detect(self, flip=True):
        self.init_detect_array()
        if flip:
            self.detect_array = np.flipud(self.detect_array)  # flip the 2d data upside down
            image_height = self.detect_array.shape[1]
        # model_input_arr = self.detect_array.reshape(*(self.detect_array.shape), 1)
        model_input_arr = self.detect_array.reshape(*(self.detect_array.shape), 1).repeat(3, axis=2)
        results = inference_detector(self.detection_model, model_input_arr)
        pred_instances = results.pred_instances.numpy()
        pred_instances = pred_instances[pred_instances.scores > self.detection_threshold]
        encode_masks = []
        bboxes = []
        for i in range(len(pred_instances)):
            m = pred_instances.masks[i]
            bbox = pred_instances.bboxes[i]
            if flip:
                m = np.flipud(m)
                bbox[[1, 3]] = image_height - bbox[[3, 1]]
            encode_masks.append(mask.encode(np.array(m[:, :, np.newaxis], order='F', dtype='uint8'))[0])
            bboxes.append(bbox)
        outputs = {'labels': pred_instances.labels.tolist(),
                   'scores': pred_instances.scores.tolist(),
                   'bboxes': bboxes,
                   'masks': encode_masks}
        if self.polar_correction or type(self.polar_correction) is dict:
            if type (self.polar_correction) is not dict:
                self.polar_correction = {}
            outputs = polar_correction(outputs, self.solar_map, **(self.polar_correction))
        if self.remove_overlap or type(self.remove_overlap) is dict:
            if type (self.remove_overlap) is not dict:
                self.remove_overlap = {}
            outputs = remove_overlap(outputs, **(self.remove_overlap))
        detect_results = []
        for i in range(len(outputs['labels'])):
            detect_results.append(SolarStructure(self, int(outputs['labels'][i]), outputs['scores'][i],
                                                 outputs['bboxes'][i], outputs['masks'][i]))
        self.detect_results = detect_results
        return outputs

    def remove_limb_brightness(self):
        coords = all_coordinates_from_map(self.solar_map)
        distance = np.sqrt((coords.Tx)**2 + (coords.Ty)**2)
        distance_rs = (distance/(self.solar_map.rsun_obs)).value
        solar_disk_mask = (distance_rs <= 1)
        phi_cos = np.sqrt(1-distance_rs[solar_disk_mask]**2)
        ratio = 4.5 - 7.3*phi_cos + 3.9*phi_cos**2
        self.detect_array[solar_disk_mask] = self.detect_array[solar_disk_mask]/ratio

    def plot(self, output="output.png"):
        fig = plt.figure()
        ax = fig.add_subplot(projection=self.solar_map)
        self.solar_map.plot(axes=ax)
        for detect_result in self.detect_results:
            decoded_mask = mask.decode(detect_result.mask)
            if detect_result.label == 0:  # coronal hole
                ax.contour(decoded_mask, colors='C0', linewidths=1)
            elif detect_result.label == 1:  # filament
                ax.contour(decoded_mask, colors='C2', linewidths=1)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output)
        plt.close("all")
        
class SolarStructure(object):
    def __init__(self, detect_info:Detection, label, score, bbox, mask) -> None:
        self.detect_info = detect_info
        self.label = label
        self.score = score
        self.bbox = bbox
        self.mask= mask
        pass

    def cal_info(self, ins_num:int, **kwargs):
        aia_map = self.detect_info.solar_map
        point_bl = aia_map.pixel_to_world(self.bbox[0]*u.pix, self.bbox[1]*u.pix)
        point_tr = aia_map.pixel_to_world(self.bbox[2]*u.pix, self.bbox[3]*u.pix)
        # ins_num = db_session.query(Instance).count()
        self.ins_id = ins_num
        current_instance = Instance(id=ins_num, score=self.score, cls=self.label,
                                    frame_id = self.detect_info.frame_id,
                                    source=self.detect_info.fits_path.split('/')[-1],
                                    time=aia_map.date.to_datetime(),
                                    box_x_min=self.bbox[0], box_x_max=self.bbox[2],
                                    box_y_min=self.bbox[1], box_y_max=self.bbox[3],
                                    box_Tx_min=point_bl.Tx.value, box_Tx_max=point_tr.Tx.value,
                                    box_Ty_min=point_bl.Ty.value, box_Ty_max=point_tr.Ty.value,
                                    mask=self.mask['counts'], **kwargs)

        decoded_mask = mask.decode(self.mask)
        contours, hierarchy = cv2.findContours(decoded_mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:  # skip error
            return
        # maskes
        topology_list = np.zeros(shape=(hierarchy.shape[1]), dtype=np.bool_)
        contour_id = 0
        topology_list[contour_id] = True
        while(hierarchy[0][contour_id][0] != -1):
            contour_id = hierarchy[0][contour_id][0]
            topology_list[contour_id] = True
        area = 0
        center_lon = 0
        center_lat = 0
        lon_max = -180
        lon_min = 180
        lat_max = -90
        lat_min = 90
        hgs_mask = np.zeros(shape=(180, 360), order='F')
        hgc_mask = np.zeros(shape=(180, 360), order='F')
        off_disk = 0
        for id, contour in enumerate(contours):
            boundary_pos = aia_map.pixel_to_world(Quantity(contour[:, 0, 0], unit='pix'),
                                                  Quantity(contour[:, 0, 1], unit='pix'))
            topology = (topology_list[id] - 0.5)*2
            # sometimes boundary is located out of the solar disk 
            off_disk_index = np.argwhere(~coordinate_is_on_solar_disk(boundary_pos)).reshape(-1)
            # if all coorinates are off disk, skip
            if np.all(~coordinate_is_on_solar_disk(boundary_pos)):
                off_disk = off_disk + 1
                continue
            # if the boundary is out of the solar disk, move it to the solar disk
            while len(off_disk_index) > 0:
                boundary_pos[off_disk_index] = SkyCoord(0.99*boundary_pos[off_disk_index].Tx, 
                                                        0.99*boundary_pos[off_disk_index].Ty, 
                                                        frame=boundary_pos.frame)
                off_disk_index = np.argwhere(~coordinate_is_on_solar_disk(boundary_pos)).reshape(-1)
            boundary_hgs = boundary_pos.transform_to(frames.HeliographicStonyhurst)
            boundary_hgc = boundary_pos.transform_to(frames.HeliographicCarrington)
            # area, center_lon, center_lat
            hgs_lon_list = boundary_hgs.lon.value
            hgs_lat_list = boundary_hgs.lat.value
            area_c, lon_c, lat_c = cal_spherical_info(hgs_lon_list, hgs_lat_list)
            area = area + area_c*topology
            if lon_c is not None and lat_c is not None:
                center_lon = center_lon + lon_c*area_c*topology
                center_lat = center_lat + lat_c*area_c*topology
            # boundary info
            if np.max(hgs_lon_list) > lon_max:
                lon_max = np.max(hgs_lon_list)
            if np.max(hgs_lat_list) > lat_max:
                lat_max = np.max(hgs_lat_list)
            if np.min(hgs_lon_list) < lon_min:
                lon_min = np.min(hgs_lon_list)
            if np.min(hgs_lat_list) < lat_min:
                lat_min = np.min(hgs_lat_list)
            # mask in HeliographicStonyhurst
            hgs_contour_mask = polygon2mask(hgs_mask.shape, np.vstack([hgs_lat_list+90, hgs_lon_list+180]).transpose())
            hgs_mask = hgs_mask + topology*hgs_contour_mask
            # mask in HeliographicCarrington
            hgc_lon_list = boundary_hgc.lon.value
            hgc_lat_list = boundary_hgc.lat.value
            delta_lon_hgs = hgs_lon_list - np.roll(hgs_lon_list, 1)
            delta_lon_hgc = hgc_lon_list - np.roll(hgc_lon_list, 1)
            carrington_ratio = np.ones_like(hgc_lon_list)
            cross_pos_list = np.nonzero(np.abs(delta_lon_hgs - delta_lon_hgc)>1)[0]
            for cross_pos in cross_pos_list:
                if delta_lon_hgc[cross_pos] < 0:
                    carrington_ratio[cross_pos:] = carrington_ratio[cross_pos:] + 1
                elif delta_lon_hgc[cross_pos] > 0:
                    carrington_ratio[cross_pos:] = carrington_ratio[cross_pos:] - 1
            hgc_lon_list = hgc_lon_list + 360*carrington_ratio
            mask_y, mask_x = polygon(hgc_lat_list+90, hgc_lon_list)
            mask_x = mask_x%360
            hgc_mask[mask_y, mask_x] += topology
        if off_disk == len(contours):
            # print("All contours are off disk!")
            return False
        # area, center_lon, center_lat
        area = np.abs(area)
        current_instance.area_ratio = area
        current_instance.area_Mm2 = area*4*np.pi*(r_sun**2)
        if area != 0:
            current_instance.average_aia_193_intensity = np.mean(aia_map.data[decoded_mask>0])/aia_map.exposure_time.value
            center_hgs_lon = center_lon/area
            center_hgs_lat = center_lat/area
            current_instance.center_hgs_lon = center_hgs_lon
            current_instance.center_hgs_lat = center_hgs_lat
            current_instance.boundary_hgs_lon_max = lon_max
            current_instance.boundary_hgs_lon_min = lon_min
            current_instance.boundary_hgs_lat_max = lat_max
            current_instance.boundary_hgs_lat_min = lat_min
            encoded_hgs_mask = mask.encode(np.asfortranarray(hgs_mask, dtype=np.uint8))
            current_instance.mask_hgs = encoded_hgs_mask['counts']
            encoded_hgc_mask = mask.encode(np.asfortranarray(hgc_mask, dtype=np.uint8))
            current_instance.mask_hgc = encoded_hgc_mask['counts']
            center_hgc = SkyCoord(center_hgs_lon, center_hgs_lat, unit='deg',
                                    obstime=current_instance.time, observer='earth',
                                    frame=frames.HeliographicStonyhurst).transform_to(frames.HeliographicCarrington)
            current_instance.center_hgc_lon = center_hgc.lon.value
            current_instance.center_hgc_lat = center_hgc.lat.value
        
        self.ins = current_instance
        return current_instance
    
    def save_db(self, db_session:Session):
        # check if the instance is already in the database
        if not hasattr(self, 'ins'):
            max_ins_id = db_session.query(func.max(Instance.id)).first()[0]
            if max_ins_id is None:
                ins_num = 1
            else:
                ins_num = max_ins_id + 1
            # skip off-disk instance
            if self.cal_info(ins_num=ins_num) is False:
                return
        db_session.add(self.ins)

