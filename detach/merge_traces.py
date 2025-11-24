import os
import cv2
import numpy as np
import tempfile
from tqdm import tqdm
from pycocotools import mask
from datetime import timedelta
from scipy.stats import linregress
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from astropy.coordinates import SkyCoord
from sunpy.map import Map
from sunpy.coordinates import frames
from sunpy.sun.constants import equatorial_radius
from aiapy.calibrate import correct_degradation
from mmengine.config import ConfigDict

from .database import Base, Instance, Trace
from .detection import cal_spherical_info


# r sun
r_sun = equatorial_radius.value/1000  # unit:km

def obj_deepcopy(obj):
    attributes = vars(obj)
    attributes = {attr: value for attr, value in attributes.items() if not attr.startswith('_')}
    new_obj = obj.__class__(**attributes)
    return new_obj

def merge_instance(query_ins_list, config, trace_id=None):
    attributes = vars(query_ins_list[0])
    ins_info = {attr: [getattr(ins, attr) for ins in query_ins_list] 
                     for attr in attributes.keys() if not attr.startswith('_')}
    if len(set(ins_info['frame_id']))>1 or len(set(ins_info['source']))>1 or \
        len(set(ins_info['time']))>1 or len(set(ins_info['cls']))>1:
        return None
    if trace_id is None:
        trace_id = ins_info['trace_id'][0]
    new_ins = Instance(id=ins_info['id'][0], trace_id=trace_id, score=np.average(ins_info['score']),
                       frame_id=ins_info['frame_id'][0], source=ins_info['source'][0],
                       time=ins_info['time'][0], cls=ins_info['cls'][0],
                       box_x_min=np.min(ins_info['box_x_min']), box_x_max=np.max(ins_info['box_x_max']),
                       box_y_min=np.min(ins_info['box_y_min']), box_y_max=np.max(ins_info['box_y_max']),
                       box_Tx_min=np.min(ins_info['box_Tx_min']), box_Tx_max=np.max(ins_info['box_Tx_max']),
                       box_Ty_min=np.min(ins_info['box_Ty_min']), box_Ty_max=np.max(ins_info['box_Ty_max']),
                       boundary_hgs_lon_min=np.min(ins_info['boundary_hgs_lon_min']),
                       boundary_hgs_lon_max=np.max(ins_info['boundary_hgs_lon_max']),
                       boundary_hgs_lat_min=np.min(ins_info['boundary_hgs_lat_min']),
                       boundary_hgs_lat_max=np.max(ins_info['boundary_hgs_lat_max']),)
    # calculate mask
    final_mask = np.zeros(shape=(4096, 4096), order='F', dtype=np.bool_)
    final_mask_hgs = np.zeros(shape=(180, 360), order='F', dtype=np.bool_)
    final_mask_hgc = np.zeros(shape=(180, 360), order='F', dtype=np.bool_)
    for ins in query_ins_list:
        final_mask = final_mask + mask.decode({'counts': ins.mask, 'size': [4096, 4096]})
        final_mask_hgs = final_mask_hgs + mask.decode({'counts': ins.mask_hgs, 'size': [180, 360]})
        final_mask_hgc = final_mask_hgc + mask.decode({'counts': ins.mask_hgc, 'size': [180, 360]})
    final_mask[final_mask>1] = 1
    final_mask_hgs[final_mask_hgs>1] = 1
    final_mask_hgc[final_mask_hgc>1] = 1
    new_ins.mask = mask.encode(final_mask)['counts']
    new_ins.mask_hgs = mask.encode(final_mask_hgs)['counts']
    new_ins.mask_hgc = mask.encode(final_mask_hgc)['counts']
    source_aia_map = Map(os.path.join(config['obs_source_dir'], new_ins.time.strftime(r"%Y/%m/%d"), new_ins.source))
    if type(config['detection_config']['recover_degrad']) in [dict, ConfigDict]:
        source_aia_map = correct_degradation(source_aia_map, **(config['detection_config']['recover_degrad']))
    elif config['detection_config']['recover_degrad']:
        source_aia_map = correct_degradation(source_aia_map)
    new_ins.average_aia_193_intensity = np.mean(source_aia_map.data[final_mask>0])/source_aia_map.exposure_time.value
    # calculate area and center
    area = 0
    center_lon = 0
    center_lat = 0
    contours, hierarchy = cv2.findContours(final_mask_hgs, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    topology_list = np.zeros(shape=(hierarchy.shape[1]), dtype=np.bool_)
    contour_id = 0
    topology_list[contour_id] = True
    while(hierarchy[0][contour_id][0] != -1):
        contour_id = hierarchy[0][contour_id][0]
        topology_list[contour_id] = True
    for id, contour in enumerate(contours):
        topology = (topology_list[id] - 0.5)*2
        hgs_lon_list = contour[:, 0, 0] - 180
        hgs_lat_list = contour[:, 0, 1] - 90
        area_c, lon_c, lat_c = cal_spherical_info(hgs_lon_list, hgs_lat_list)
        area = area + area_c*topology
        if lon_c is not None and lat_c is not None:
            center_lon = center_lon + lon_c*area_c*topology
            center_lat = center_lat + lat_c*area_c*topology
    if area != 0:
        center_hgs_lon = center_lon/area
        center_hgs_lat = center_lat/area
        new_ins.area_ratio = area
        new_ins.area_km2 = area*4*np.pi*(r_sun**2)
        new_ins.center_hgs_lon = center_hgs_lon
        new_ins.center_hgs_lat = center_hgs_lat
        center_hgc = SkyCoord(center_hgs_lon, center_hgs_lat, unit='deg',
                              obstime=new_ins.time, observer='earth',
                              frame=frames.HeliographicStonyhurst).transform_to(frames.HeliographicCarrington)
        new_ins.center_hgc_lon = center_hgc.lon.value
        new_ins.center_hgc_lat = center_hgc.lat.value
    return new_ins

def merge_trace_from_ins(ins_list, config, trace_id=0):
    # TODO merge trace code
    time_list = np.array([ins.time for ins in ins_list])
    unique_time = sorted(list(set(time_list)))
    output_ins_list = []
    for t in unique_time:
        ins_mask = (time_list == t)
        pos = np.flatnonzero(ins_mask)
        ins_num = len(pos)
        if ins_num == 1:
            output_ins_list.append(obj_deepcopy(ins_list[pos[0]]))
        else:
            to_merge_ins_list = [ins_list[i] for i in pos]
            new_ins = merge_instance(to_merge_ins_list, config, trace_id)
            output_ins_list.append(new_ins)
    aia_intensity_list = np.zeros(shape=(len(output_ins_list)))
    area_ratio_list = np.zeros(shape=(len(output_ins_list)))
    area_km2_list = np.zeros(shape=(len(output_ins_list)))
    boundary_hgs_y_min_list = np.zeros(shape=(len(output_ins_list)))
    boundary_hgs_y_max_list = np.zeros(shape=(len(output_ins_list)))
    center_lon_list = np.zeros(shape=(len(output_ins_list)))
    center_lat_list = np.zeros(shape=(len(output_ins_list)))
    score_list = np.zeros(shape=(len(output_ins_list)))
    for i, ins in enumerate(output_ins_list):
        ins.trace_id = trace_id
        aia_intensity_list[i] = ins.average_aia_193_intensity
        area_ratio_list[i] = ins.area_ratio
        area_km2_list[i] = ins.area_km2
        boundary_hgs_y_min_list[i] = ins.boundary_hgs_lat_min
        boundary_hgs_y_max_list[i] = ins.boundary_hgs_lat_max
        center_lon_list[i] = ins.center_hgs_lon
        center_lat_list[i] = ins.center_hgs_lat
        score_list[i] = ins.score

    new_trace = Trace(id=trace_id, start_time=output_ins_list[0].time,
                      end_time=output_ins_list[-1].time, instance_num=len(output_ins_list),
                      average_score=np.average(score_list),
                      cls=output_ins_list[0].cls, average_latitude=np.average(center_lat_list),
                      latitude_std=np.std(center_lat_list),
                      average_aia_193_intensity=np.average(aia_intensity_list),
                      aia_193_intensity_std=np.std(aia_intensity_list),
                      average_area_ratio=np.average(area_ratio_list),
                      area_ratio_std=np.std(area_ratio_list),
                      average_area_km2=np.average(area_km2_list),
                      area_km2_std=np.std(area_km2_list),
                      boundary_hgs_y_min=np.min(boundary_hgs_y_min_list),
                      boundary_hgs_y_max=np.max(boundary_hgs_y_max_list),
                      boundary_hgs_y_min_err=np.std(boundary_hgs_y_min_list),
                      boundary_hgs_y_max_err=np.std(boundary_hgs_y_max_list))
    time_day_list = [(t - unique_time[0])/timedelta(days=1) for t in unique_time]  # time delta in days
    cycle_num = 0
    updated_center_lon = []
    for i in range(len(unique_time)):
        if (time_day_list[i] - time_day_list[i-1])>8 and (center_lon_list[i]<center_lon_list[i-1]):
            cycle_num += 1
        elif (time_day_list[i] - time_day_list[i-1])>25:
            cycle_num += 1
        updated_center_lon.append(center_lon_list[i]+cycle_num*360)
    new_trace.cycle = cycle_num+1
    if len(unique_time) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(time_day_list, center_lat_list)
        new_trace.hgs_latitude_speed = slope
        new_trace.hgs_latitude_speed_r = r_value
        new_trace.hgs_latitude_speed_std = std_err
        slope, intercept, r_value, p_value, std_err = linregress(time_day_list, updated_center_lon)
        new_trace.hgs_longitude_speed = slope
        new_trace.hgs_longitude_speed_r = r_value
        new_trace.hgs_longitude_speed_std = std_err
    return new_trace, output_ins_list

def cal_trace_iou(tr1, tr2):
    # calculate the distance between two traces
    if tr1.first().cls != tr2.first().cls:
        return 0
    lat_list = np.arange(-90, 90, 1).reshape(180, 1)

    info_tr1 = tr1.with_entities(Instance.mask_hgc, Instance.score).all()
    rle_list_1 = [{'counts': r[0], 'size': [180, 360]} for r in info_tr1]
    score_list_1 = np.array([r[1] for r in info_tr1]).reshape((1, 1, -1))
    hgc_mask_1 = mask.decode(rle_list_1)
    avg_mask_1 = np.average(hgc_mask_1*score_list_1, axis=2)*np.cos(lat_list*np.pi/180)

    info_tr2 = tr2.with_entities(Instance.mask_hgc, Instance.score).all()
    rle_list_2 = [{'counts': r[0], 'size': [180, 360]} for r in info_tr2]
    score_list_2 = np.array([r[1] for r in info_tr2]).reshape((1, 1, -1))
    hgc_mask_2 = mask.decode(rle_list_2)
    avg_mask_2 = np.average(hgc_mask_2*score_list_2, axis=2)*np.cos(lat_list*np.pi/180)

    if np.sum((avg_mask_1>0)&(avg_mask_2>0)) == 0:
        return 0
    intersection = np.sum(np.where(avg_mask_1<avg_mask_2, avg_mask_1, avg_mask_2))
    union = np.sum(np.where(avg_mask_1>avg_mask_2, avg_mask_1, avg_mask_2))
    iou = intersection/union
    return iou

def cal_trace_distance(tr1, tr2):
    # calculate the distance between two traces
    if tr1.first().cls != tr2.first().cls:
        return 0
    info_tr1 = tr1.with_entities(Instance.mask_hgc, Instance.score).all()
    rle_list_1 = [{'counts': r[0], 'size': [180, 360]} for r in info_tr1]
    score_list_1 = np.array([r[1] for r in info_tr1]).reshape((1, 1, -1))
    hgc_mask_1 = mask.decode(rle_list_1)
    avg_mask_1 = np.average(hgc_mask_1*score_list_1, axis=2)

    info_tr2 = tr2.with_entities(Instance.mask_hgc, Instance.score).all()
    rle_list_2 = [{'counts': r[0], 'size': [180, 360]} for r in info_tr2]
    score_list_2 = np.array([r[1] for r in info_tr2]).reshape((1, 1, -1))
    hgc_mask_2 = mask.decode(rle_list_2)
    avg_mask_2 = np.average(hgc_mask_2*score_list_2, axis=2)

    if np.sum((avg_mask_1>0)&(avg_mask_2>0)) == 0:
        return 0
    lat_list = np.arange(-90, 90, 1).reshape(180, 1)
    mul_mask = avg_mask_1*avg_mask_2*np.cos(lat_list*np.pi/180)
    distance = np.average(mul_mask[(avg_mask_1>0)|(avg_mask_2>0)])
    return distance

def update_db(config, buffer_db_path=None, metrics='distance', merge_threshold=0.1, commit_interval=10):
    # generate buffer database path
    db_path = config['database_path']
    if buffer_db_path is None:
        # dir_path = os.path.dirname(db_path)
        dir_path = os.getcwd()
        buffer_db_path = tempfile.mktemp(dir=dir_path, prefix="detach_buffer_", suffix=".db")
        buffer_db_path = "sqlite:///"+buffer_db_path
    # calculate metrics
    if metrics == 'distance':
        cal_func = cal_trace_distance
    elif metrics == "iou":
        cal_func = cal_trace_iou
    else:
        print(f"function {metrics} not supported")
        return

    # old buffer database
    engine0 = create_engine(db_path)
    DBSession0 = sessionmaker(bind=engine0)
    Base.metadata.create_all(engine0)
    session_old = DBSession0()
    # new generated database
    engine1 = create_engine(buffer_db_path)
    DBSession1 = sessionmaker(bind=engine1)
    Base.metadata.create_all(engine1)
    session_new = DBSession1()
    
    # merge traces
    all_traces = session_old.query(Trace).all()
    for tr_num, cr_tr in enumerate(tqdm(all_traces, desc="Merging traces")):
        pr_traces = session_new.query(Trace).filter(Trace.end_time>cr_tr.start_time-timedelta(days=27)).all()
        curr_ins_list = session_old.query(Instance).filter(Instance.trace_id==cr_tr.id)
        merge_flag = False
        to_merge_ins_list = []
        del_trace_list = []
        new_trace_id = cr_tr.id
        for pr_tr in pr_traces:
            prev_ins_list = session_new.query(Instance).filter(Instance.trace_id==pr_tr.id)
            tr_distance = cal_func(curr_ins_list, prev_ins_list)
            if tr_distance > merge_threshold:
                to_merge_ins_list = to_merge_ins_list + prev_ins_list.all()
                del_trace_list.append(pr_tr)
                if pr_tr.id < new_trace_id:
                    new_trace_id = pr_tr.id
                merge_flag = True
        # double check those who merge more than 2 traces
        if merge_flag and len(del_trace_list)>1:
            cr_tr_num = curr_ins_list.count()
            cr_tr_score = curr_ins_list.with_entities(func.avg(Instance.score)).first()
            pr_tr_num_list = []
            pr_tr_score_list = []
            for pr_tr in del_trace_list:
                prev_ins_list = session_new.query(Instance).filter(Instance.trace_id==pr_tr.id)
                pr_tr_num = prev_ins_list.count()
                pr_tr_score = prev_ins_list.with_entities(func.avg(Instance.score)).first()
                pr_tr_num_list.append(pr_tr_num)
                pr_tr_score_list.append(pr_tr_score)
            # conditions for those not to merge
            if cr_tr_num < np.min(pr_tr_num_list) or cr_tr_score < np.max(pr_tr_score_list):
                merge_flag = False

        # add traces and instance object to new database
        if merge_flag:
            new_tr, new_ins_list = merge_trace_from_ins(to_merge_ins_list+curr_ins_list.all(), config, trace_id=new_trace_id)
            for prev_ins in to_merge_ins_list:
                session_new.delete(prev_ins)
            for new_ins in new_ins_list:
                session_new.add(new_ins)
            for pr_tr in del_trace_list:
                session_new.delete(pr_tr)
            session_new.add(new_tr)
        else:
            for ins in curr_ins_list.all():
                session_new.add(obj_deepcopy(ins))
            session_new.add(obj_deepcopy(cr_tr))
        # commit to database
        if tr_num % commit_interval == 0:
            session_new.commit()
    session_new.commit()
    return buffer_db_path
