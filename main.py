import os
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import Counter
from datetime import datetime, timedelta
from pycocotools import mask
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
from scipy.stats import linregress
from astropy.units import Quantity
import astropy.units as u
from astropy.io import fits
from sunpy.coordinates import frames
from sunpy.map import Map
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

from mmengine.config import Config
from mmdet.apis import init_detector
from detach import Detection
from detach.database import Base, Instance, Trace
from detach.tracker import Tracker
from detach.nn_matching import NearestNeighborDistanceMetric
from detach import merge_traces

import logging
# ignore Info output of the reproject mode
logging.getLogger('reproject').setLevel(logging.WARNING)

import warnings
# ignore warning when calculating average value when the array is empty
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
# ignore warning when area is zero
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
# ignore warning when calculating variance when the array only has one value
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice.")


ch_color_list = ['cyan', 'blue', 'darkturquoise', 'cadetblue', 'deepskyblue', 'lightskyblue', 'steelblue', 'darkblue', 'royalblue', 'mediumblue', 'blueviolet', 'mediumslateblue']
fi_color_list = ['yellowgreen', 'greenyellow', 'lawngreen', 'green', 'limegreen', 'lime', 'springgreen']


class DETACH(object):
    def __init__(self, config_path) -> None:
        config = Config.fromfile(config_path)
        self.cfg = config
        # initial database
        engine = create_engine(config['database_path'])
        DBSession = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        self.engine = engine
        self.session = DBSession()
        # detection model
        self.detection_model = init_detector(config_path, config['detection_checkpoint'], device=config['device'])
        # tracking
        metric = NearestNeighborDistanceMetric(metric=config['tracking_config']['distance_metric'],
                                               matching_threshold=config['tracking_config']['max_dist'],
                                               budget=config['tracking_config']['nn_budget'],
                                               reduction=config['tracking_config']['reduction'],
                                               batch_size=config['tracking_config']['batch_size'])
        self.tracker = Tracker(metric, time_interval=timedelta(**(config['time_interval'])),
                               max_iou_distance=config['tracking_config']['max_iou_distance'],
                               max_age=config['tracking_config']['max_age'],
                               n_init=config['tracking_config']['n_init'])
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self, commit_interval=10, save_img=False, save_dir='output', track=True):
        start_time = datetime(**(self.cfg['start_time']))
        end_time = datetime(**(self.cfg['end_time']))
        interval = timedelta(**(self.cfg['time_interval']))
        # calculate time list
        time_list = []
        buffer_t = start_time
        while buffer_t < end_time:
            time_list.append(buffer_t)
            buffer_t = buffer_t + interval
        # detect the detailed boundary
        self.detect(time_list, commit_interval=commit_interval, track=track)
        if track:
            tracking_config = self.cfg['tracking_config']
            self.add_trace_table(commit_interval=commit_interval)
            self.cal_trace_info(commit_interval=commit_interval)
            if 'merge_traces' in tracking_config and isinstance(tracking_config, dict):
                buffer_db_path = merge_traces.update_db(self.cfg, **(tracking_config['merge_traces']))
                self.update_merged_db(buffer_db_path)
                # remove buffer sqlite database
                db_file = buffer_db_path.replace("sqlite:///", "")
                if os.path.exists(db_file):
                    os.remove(db_file)
        # magnetic info
        self.add_magnetic_info(commit_interval=commit_interval, hmi_source_dir=self.cfg['hmi_source_dir'])
        if track:
            self.cal_trace_mag_info(commit_interval=commit_interval)
        if save_img:
            os.makedirs(save_dir, exist_ok=True)
            # use agg backend to avoid memory leak
            matplotlib.use('agg')
            for t in (pbar := tqdm(time_list)):
                pbar.set_description("Output images... current time: " + t.isoformat())
                file_list1 = glob(os.path.join(self.cfg['obs_source_dir'],
                                              t.strftime(r"%Y/%m/%d/aia_lev1_193a_%Y_%m_%dt%H_%M_*_image_lev1.fits")))
                file_list2 = glob(os.path.join(self.cfg['obs_source_dir'],
                                               t.strftime(r"%Y/%m/%d/aia.lev1.193A_%Y_%m_%dT%H_%M_*.image_lev1.fits")))
                file_list3 = glob(os.path.join(self.cfg['obs_source_dir'],
                                              t.strftime(r"%Y/%m/%d/aia.lev1_euv_12s.%Y-%m-%dT%H%M*Z.193.image_lev1.fits")))
                file_list = file_list1+file_list2+file_list3
                self.tracker.predict()
                if len(file_list) == 0:
                    continue
                file_path = sorted(file_list)[0]
                detector = Detection(file_path, detection_model=None, **(self.cfg['detection_config']))
                if detector.solar_map.meta["crpix1"] == 'nan' or\
                    np.abs(detector.solar_map.meta["crpix1"]-2048)>1000 or\
                        np.abs(detector.solar_map.meta["crpix2"]-2048)>1000: 
                    continue
                detector.init_detect_array()
                ins_list = self.session.query(Instance).filter(Instance.source == file_path.split('/')[-1]).all()
                save_path = os.path.join(save_dir, t.strftime(r"detach_aia_193_%Y_%m_%dT%H_%M_%S")+'.png')
                self.save_img(detector, ins_list, save_path=save_path)

    def detect(self, time_list, commit_interval=10, track=True):
        # prediction & save to database
        for frame_id, t in enumerate(pbar := tqdm(time_list)):
            pbar.set_description("Detecting... current time: " + t.isoformat())
            file_list1 = glob(os.path.join(self.cfg['obs_source_dir'],
                                            t.strftime(r"%Y/%m/%d/aia_lev1_193a_%Y_%m_%dt%H_%M_*_image_lev1.fits")))
            file_list2 = glob(os.path.join(self.cfg['obs_source_dir'],
                                            t.strftime(r"%Y/%m/%d/aia.lev1.193A_%Y_%m_%dT%H_%M_*.image_lev1.fits")))
            file_list3 = glob(os.path.join(self.cfg['obs_source_dir'],
                                            t.strftime(r"%Y/%m/%d/aia.lev1_euv_12s.%Y-%m-%dT%H%M*Z.193.image_lev1.fits")))
            file_list = file_list1+file_list2+file_list3
            if len(file_list) == 0:
                continue
            file_path = sorted(file_list)[0]
            detector = Detection(file_path, detection_model=self.detection_model, frame_id=frame_id, **(self.cfg['detection_config']))
            # skip data with missing header(e.g. 2020.2.24 lunar transit)
            # skip data far away from center(e.g. 2011.4.14, 2021.4.29)
            if detector.solar_map.meta["crpix1"] == 'nan' or\
                np.abs(detector.solar_map.meta["crpix1"]-2048)>1000 or\
                    np.abs(detector.solar_map.meta["crpix2"]-2048)>1000: 
                continue
            outputs = detector.detect()
            for result in detector.detect_results:
                result.save_db(db_session=self.session)
            if track:
                self.tracker.predict()
                ins_list = self.session.query(Instance).filter(Instance.source == file_path.split('/')[-1]).all()
                self.tracker.update(ins_list)
            if pbar.n % commit_interval == 0:
                self.session.commit()
        self.session.commit()
    
    def add_magnetic_info(self, commit_interval=10, hmi_source_dir="/media/ExtHDD/data/HMI-b-ptr/"):
        info_list = self.session.query(Instance.time, Instance.source).all()
        for info in (pbar := tqdm(sorted(set(info_list)), desc="Calculating magnetic field info")):
            aia_file_path = os.path.join(self.cfg['obs_source_dir'], info[0].strftime(r"%Y/%m/%d/"), info[1])
            hmi_file_path0 = os.path.join(hmi_source_dir, info[0].strftime(r"%Y/%m/%d/"), f"hmi.b_720s.{info[0].strftime(r'%Y%m%d_%H%M')}00_TAI.Br.fits")
            hmi_file_path1 = os.path.join(hmi_source_dir, info[0].strftime(r"%Y/%m/%d/"), f"hmi.B_720s.{info[0].strftime(r'%Y%m%d_%H%M')}00_TAI.Br.fits")
            if os.path.exists(hmi_file_path0):
                hmi_file_path = hmi_file_path0
            elif os.path.exists(hmi_file_path1):
                hmi_file_path = hmi_file_path1
            else:
                continue
            try:
                obs = fits.open(hmi_file_path)[1]
                obs.header['BUNIT'] = 'Gs'
                hmi_map = Map(obs.data, obs.header)
                # hmi_map = Map(hmi_file_path)
            except Exception as e:
                print(e)
                continue
            # calculate radial magnetic field
            aia_map = Map(aia_file_path)
            reproject_hmi_map = hmi_map.reproject_to(aia_map.wcs)
            ins_list = self.session.query(Instance).filter(Instance.source==info[1]).all()
            for ins in ins_list:
                decoded_mask = mask.decode({'counts': ins.mask, 'size': [4096, 4096]})
                y_inner, x_inner = np.where(decoded_mask)
                world_pos = aia_map.pixel_to_world(Quantity(x_inner, unit='pix'),
                                                   Quantity(y_inner, unit='pix')).transform_to(frames.HeliocentricEarthEcliptic)
                b_r = reproject_hmi_map.data[y_inner, x_inner]
                b_r[np.isnan(b_r)] = 0
                ds0 = (((aia_map.scale[0]*1*u.pix).to(u.rad)*aia_map.meta['dsun_obs']/1e6)**2).value  # area for a pixel, Unit: Mm^2
                d_area = (1/np.cos(world_pos.lon.value*np.pi/180)/np.cos(world_pos.lat.value*np.pi/180))
                d_area[np.isnan(d_area)] = 0
                # 1 G Mm^2 = 1e8 Wb; use the same units for all fluxes.
                flux = np.sum(b_r*d_area*ds0)*1e8
                flux_pos = np.sum((b_r*d_area)[b_r > 0])*ds0*1e8
                flux_neg = np.sum((b_r*d_area)[b_r < 0])*ds0*1e8
                area_pos = np.sum(d_area[b_r > 0])*ds0
                area_neg = np.sum(d_area[b_r < 0])*ds0
                ins.flux = flux
                ins.flux_positive = flux_pos
                ins.flux_negative = flux_neg
                ins.area_positive = area_pos
                ins.area_negative = area_neg
                ins.B_average = flux/ins.area_Mm2/1e8
                ins.B_pos_avg = flux_pos/area_pos/1e8
                ins.B_neg_avg = flux_neg/area_neg/1e8
                ins.B_abs_average = (flux_pos-flux_neg)/(area_pos+area_neg)/1e8
                ins.unipolarity_degree = (ins.B_abs_average - abs(ins.B_average))/(ins.B_abs_average)
                if flux != 0:
                    ins.flux_percent_unbalanced = abs(flux)/(flux_pos - flux_neg)
            if pbar.n % commit_interval == 0:
                self.session.commit()
        self.session.commit()
    
    def add_trace_table(self, commit_interval=10):
        trace_id_min, trace_id_max = self.session.query(func.min(Instance.trace_id), func.max(Instance.trace_id)).first()
        for trace_id in (pbar := tqdm(range(trace_id_min, trace_id_max+1, 1), desc="Adding trace table")):
            ins_list = self.session.query(Instance).filter(Instance.trace_id == trace_id)
            ins_num = ins_list.count()
            start_t, end_t = ins_list.with_entities(func.min(Instance.time), func.max(Instance.time)).first()
            trace = Trace(id=trace_id, start_time=start_t, end_time=end_t, instance_num=ins_num)
            self.session.add(trace)
            if pbar.n % commit_interval == 0:
                self.session.commit()
        self.session.commit()
    
    def cal_trace_info(self, commit_interval=20):
        traces = self.session.query(Trace).all()
        for trace in (pbar := tqdm(traces, desc="Calculating trace information")):
            ins_list = self.session.query(Instance).filter(Instance.trace_id == trace.id)
            # class
            cls_list = ins_list.with_entities(Instance.cls).all()
            c = Counter(cls_list)
            if len(c) > 1:
                ch_num = c[(0,)]
                fi_num = c[(1,)]
                if ch_num >= fi_num:
                    trace.cls = 0
                    trace.comment = "contain possible filaments"
                else:
                    trace.cls = 1
                    trace.comment = "contain possible coronal holes"
            else:
                trace.cls = list(c.keys())[0][0]
            # area
            basic_info = ins_list.with_entities(Instance.average_aia_193_intensity, Instance.score,
                                                Instance.area_ratio, Instance.area_Mm2).all()
            basic_info = np.array(basic_info)
            trace.average_aia_193_intensity = float(np.mean(basic_info[:, 0]))
            trace.aia_193_intensity_std = float(np.std(basic_info[:, 0]))
            trace.average_score = float(np.mean(basic_info[:, 1]))
            trace.average_area_ratio = float(np.mean(basic_info[:, 2]))
            trace.area_ratio_std = float(np.std(basic_info[:, 2]))
            trace.average_area_Mm2 = float(np.mean(basic_info[:, 3]))
            trace.area_Mm2_std = float(np.std(basic_info[:, 3]))
            # boundary y
            boundary_hgs_y_info = ins_list.with_entities(Instance.boundary_hgs_lat_min,
                                                         Instance.boundary_hgs_lat_max).all()
            boundary_hgs_y_info = np.array(boundary_hgs_y_info)
            trace.boundary_hgs_y_min = float(np.min(boundary_hgs_y_info[:, 0]))
            trace.boundary_hgs_y_max = float(np.max(boundary_hgs_y_info[:, 1]))
            trace.boundary_hgs_y_min_err = float(np.std(boundary_hgs_y_info[:, 0]))
            trace.boundary_hgs_y_max_err = float(np.std(boundary_hgs_y_info[:, 1]))
            # center
            center_info = ins_list.with_entities(Instance.center_hgs_lon, Instance.center_hgs_lat).all()
            center_info = np.array(center_info)
            center_lon = center_info[:, 0].tolist()
            center_lat = center_info[:, 1].tolist()
            trace.average_latitude = float(np.mean(center_info[:, 1]))
            trace.latitude_std = float(np.std(center_info[:, 1]))
            # cycle num & speed
            time_info = ins_list.with_entities(Instance.time).all()
            time_list = [(t[0] - time_info[0][0])/timedelta(days=1) for t in time_info]  # time delta in days
            cycle_num = 0
            updated_center_lon = []
            for i in range(len(time_list)):
                if (time_list[i] - time_list[i-1])>8 and (center_lon[i]<center_lon[i-1]):
                    cycle_num += 1
                elif (time_list[i] - time_list[i-1])>25:
                    cycle_num += 1
                updated_center_lon.append(center_lon[i]+cycle_num*360)
            trace.cycle = cycle_num+1
            if len(time_list) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(time_list, center_lat)
                trace.hgs_latitude_speed = slope
                trace.hgs_latitude_speed_r = r_value
                trace.hgs_latitude_speed_std = std_err
                slope, intercept, r_value, p_value, std_err = linregress(time_list, updated_center_lon)
                trace.hgs_longitude_speed = slope
                trace.hgs_longitude_speed_r = r_value
                trace.hgs_longitude_speed_std = std_err
            if pbar.n % commit_interval == 0:
                self.session.commit()
        self.session.commit()
            
    def cal_trace_mag_info(self, commit_interval=20):
        traces = self.session.query(Trace).all()
        for trace in (pbar := tqdm(traces, desc="Calculating trace magnetic information")):
            ins_list = self.session.query(Instance).filter(Instance.trace_id == trace.id)
            # flux 
            flux_info = ins_list.with_entities(Instance.flux, Instance.flux_positive, Instance.flux_negative,
                                               Instance.area_positive, Instance.area_negative,
                                               Instance.B_average, Instance.B_pos_avg, Instance.B_neg_avg,
                                               Instance.B_abs_average, Instance.unipolarity_degree).all()
            flux_info = np.array(flux_info, dtype=float)
            trace.flux = float(np.nanmean(flux_info[:, 0]))
            trace.flux_relative_err = float(np.nanstd(flux_info[:, 0]))
            trace.flux_pos = float(np.nanmean(flux_info[:, 1]))
            trace.flux_pos_relative_err = float(np.nanstd(flux_info[:, 1]))
            trace.flux_neg = float(np.nanmean(flux_info[:, 2]))
            trace.flux_neg_relative_err = float(np.nanstd(flux_info[:, 2]))

            trace.area_positive = float(np.nanmean(flux_info[:, 3]))
            trace.area_negative = float(np.nanmean(flux_info[:, 4]))

            trace.B_average = float(np.nanmean(flux_info[:, 5]))
            trace.B_err = float(np.nanstd(flux_info[:, 5]))
            trace.B_pos = float(np.nanmean(flux_info[:, 6]))
            trace.B_pos_err = float(np.nanstd(flux_info[:, 6]))
            trace.B_neg = float(np.nanmean(flux_info[:, 7]))
            trace.B_neg_err = float(np.nanstd(flux_info[:, 7]))
            trace.B_abs_average = float(np.nanmean(flux_info[:, 8]))
            trace.B_abs_err = float(np.nanstd(flux_info[:, 8]))
            trace.unipolarity_degree = float(np.nanmean(flux_info[:, 9]))
            trace.uni_deg_err = float(np.nanstd(flux_info[:, 9]))
            if pbar.n % commit_interval == 0:
                self.session.commit()
        self.session.commit()

    def update_merged_db(self, merged_db_path):
        # remove old result first
        Base.metadata.reflect(bind=self.engine)
        Base.metadata.drop_all(bind=self.engine)
        db_path = self.cfg['database_path']
        with self.engine.connect() as conn:
            if db_path.startswith("postgresql"):
                conn.execute(text("VACUUM FULL"))
            elif db_path.startswith("sqlite"):
                conn.execute(text("VACUUM"))
            elif db_path.startswith("mysql"):
                conn.execute(text("OPTIMIZE TABLE instance"))
                conn.execute(text("OPTIMIZE TABLE trace"))
        # new database
        Base.metadata.create_all(bind=self.engine)
        # add merged result to database
        buffer_engine = create_engine(merged_db_path)
        DBSession_m = sessionmaker(bind=buffer_engine)
        session_m = DBSession_m()
        # calculate instance and trace id to keep continous
        max_ins_id, max_trace_id = session_m.query(func.max(Instance.id), func.max(Instance.trace_id)).first()
        new_ins_id = 1
        new_trace_id = 1
        # instance
        for old_ins_id in tqdm(range(max_ins_id+1), desc="Serialization detection object: "):
            ins = session_m.get(Instance, old_ins_id)
            if ins is None:
                continue
            else:
                new_ins = merge_traces.obj_deepcopy(ins)
                new_ins.id = new_ins_id
                self.session.add(new_ins)
                new_ins_id += 1
        self.session.commit()
        # trace
        for old_trace_id in tqdm(range(max_trace_id+1), desc="Serialization tracing ids: "):
            ins_list = self.session.query(Instance).filter(Instance.trace_id == old_trace_id).all()
            if len(ins_list) == 0:
                continue
            for ins in ins_list:
                ins.trace_id = new_trace_id
            old_trace = session_m.get(Trace, old_trace_id)
            new_trace = merge_traces.obj_deepcopy(old_trace)
            new_trace.id = new_trace_id
            self.session.add(new_trace)
            new_trace_id += 1
        self.session.commit()

    def save_img(self, detector, ins_list=[], save_path="output/output.png"):
        fig = plt.figure()
        ax = fig.add_subplot(projection=detector.solar_map)
        detector.solar_map.plot(axes=ax)
        for ins in ins_list:
            label=ins.trace_id
            if ins.cls == 0:
                c = 'C0' if label is None else ch_color_list[label%len(ch_color_list)]
                name= 'CH'
            else:
                c = 'C2' if label is None else fi_color_list[label%len(fi_color_list)]
                name= 'FI'
            decoded_mask = mask.decode({'counts': ins.mask, 'size': [4096, 4096]})
            bbox_x = [ins.box_x_min, ins.box_x_max, ins.box_x_max, ins.box_x_min, ins.box_x_min]
            bbox_y = [ins.box_y_min, ins.box_y_min, ins.box_y_max, ins.box_y_max, ins.box_y_min]
            ax.contourf(decoded_mask, levels=1, colors=[(1, 1, 1, 0), (*(colors.to_rgb(c)), 0.4)])
            ax.plot(bbox_x, bbox_y, linewidth=1, color=c)
            ax.text(bbox_x[0], bbox_y[2], name+'-'+str(label), horizontalalignment='left', verticalalignment='top', color='white')
        fig.savefig(save_path)
        plt.close("all")
    
    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h


if __name__ == "__main__":
    detach_object = DETACH(config_path="checkpoints/DETACH_config.py")
    detach_object.run(save_img=False, commit_interval=10, track=True)