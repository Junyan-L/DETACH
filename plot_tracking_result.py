import os
import matplotlib
import matplotlib.pyplot as plt
from sunpy.map import Map
from pycocotools import mask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from detach.database import Base, Instance
from mmengine.config import Config

def main(ch_id, config_path, output_dir="vis"):
    # use agg backend to avoid memory leak
    matplotlib.use('agg')
    output_dir = output_dir+"-"+str(ch_id)
    os.makedirs(output_dir, exist_ok=True)

    config = Config.fromfile(config_path)
    # initial database
    engine = create_engine(config['database_path'])
    DBSession = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    session = DBSession()
    ins_list = session.query(Instance).filter(Instance.ch_id_origin == ch_id).all()
    for ins in ins_list:
        # solar_map = Map("")
        # fig = plt.figure()
        # ax = fig.add_subplot(projection=solar_map)
        # solar_map.plot(axes=ax)
        m = ins.mask
        decoded_mask = mask.decode({"size": [4096, 4096], "counts": ins.mask})
        plt.contour(decoded_mask, colors='C0', linewidths=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(output_dir, (ins.source)[:-5]+".png"))
        plt.close("all")

if __name__ == "__main__":
    main(2, config_path="checkpoints/DETACH_config.py")