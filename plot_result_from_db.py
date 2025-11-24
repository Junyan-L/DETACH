import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime, timedelta
from sunpy.map import Map
from pycocotools import mask
from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker
from detach.database import Base, Instance
from mmengine.config import Config


ch_color_list = ['cyan', 'blue', 'darkturquoise', 'cadetblue', 'deepskyblue', 'lightskyblue', 'steelblue', 'darkblue', 'royalblue', 'mediumblue', 'blueviolet', 'mediumslateblue']
fi_color_list = ['yellowgreen', 'greenyellow', 'lawngreen', 'green', 'limegreen', 'lime', 'springgreen']


def plot_single_image(source_obstime, session, output_dir="vis_pdf", output_format="pdf"):
    ins_list = session.query(Instance).filter(and_(Instance.time<source_obstime+timedelta(hours=1),
                                                   Instance.time>source_obstime-timedelta(hours=1))).all()
    if len(ins_list) == 0:
        print(source_obstime.isoformat()+" No detection data is found!")
        return
    obs_time = ins_list[0].time
    fits_source = obs_time.strftime(rf"/media/ExtHDD/data/AIA-193/%Y/%m/%d/{ins_list[0].source}")
    solar_map = Map(fits_source)
    fig = plt.figure(figsize=(6, 5))
    fig.tight_layout()
    plt.gcf().subplots_adjust(left=0.16, bottom=0, top=1, right=0.98)
    ax = fig.add_subplot(projection=solar_map)
    solar_map.plot(axes=ax)
    for ins in ins_list:
        label=ins.trace_id
        if ins.cls == 0:
            c = 'C0' if label is None else ch_color_list[label%len(ch_color_list)]
            name= 'CH'
        else:
            c = 'C2' if label is None else fi_color_list[label%len(fi_color_list)]
            name= 'Fil'
        decoded_mask = mask.decode({'counts': ins.mask, 'size': [4096, 4096]})
        bbox_x = [ins.box_x_min, ins.box_x_max, ins.box_x_max, ins.box_x_min, ins.box_x_min]
        bbox_y = [ins.box_y_min, ins.box_y_min, ins.box_y_max, ins.box_y_max, ins.box_y_min]
        ax.contourf(decoded_mask, levels=1, colors=[(1, 1, 1, 0), (*(colors.to_rgb(c)), 0.4)])
        ax.plot(bbox_x, bbox_y, linewidth=1, color=c)
        ax.text(bbox_x[0], bbox_y[2], name+'-'+str(label), horizontalalignment='left', verticalalignment='top', color='white')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(shrink=0.8)
    # plt.savefig(os.path.join(output_dir, (ins.source)[:-5]+".pdf"))
    plt.savefig(os.path.join(output_dir, source_obstime.strftime(r"detach_aia_193_%Y_%m_%dT%H_%M_%S."+output_format)))
    plt.close("all")

def main(config_path, start_time, end_time=None, delta_interval=timedelta(hours=12),
         output_dir="vis_pdf", output_format="pdf"):
    # use agg backend to avoid memory leak
    matplotlib.use('agg')
    os.makedirs(output_dir, exist_ok=True)

    config = Config.fromfile(config_path)
    # initial database
    engine = create_engine(config['database_path'])
    DBSession = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    session = DBSession()
    if end_time is None:
        plot_single_image(start_time, session, output_dir=output_dir, output_format=output_format)
        return
    t = start_time
    while t < end_time:
        plot_single_image(t, session, output_dir=output_dir, output_format=output_format)
        t = t + delta_interval
    

if __name__ == "__main__":
    main(config_path="checkpoints/DETACH_config.py",
         start_time=datetime(2010, 5, 13),
         end_time=datetime(2010, 7, 1),
         delta_interval=timedelta(hours=12),
         output_dir='output', output_format='png')