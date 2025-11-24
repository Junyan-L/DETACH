import os
import glob
import matplotlib
# use agg backend to avoid memory leak
matplotlib.use('agg')
from tqdm import tqdm
from mmdet.apis import init_detector
from detach import Detection
from mmengine.config import Config
import matplotlib.pyplot as plt
from pycocotools import mask


def plot_all(img_list, config_path="config.py", output_dir="vis", output_format='pdf'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose to use a config
    config = Config.fromfile(config_path)
    detection_model = init_detector(config=config_path,
                                    checkpoint=config['detection_checkpoint'],
                                    device=config['device'])

    fig = plt.figure(figsize=(18, 24))
    for total_id, img in enumerate(pbar := tqdm(img_list)):
        out_img_name = img.split('/')[-1]
        pbar.set_description("current: " + out_img_name)
        detector = Detection(img, detection_model=detection_model, **(config['detection_config']))
        detector.detect()

        current_id = total_id % 15
        ax = fig.add_subplot(5, 3, current_id+1, projection=detector.solar_map)
        detector.solar_map.plot(axes=ax)
        for detect_result in detector.detect_results:
            decoded_mask = mask.decode(detect_result.mask)
            if detect_result.label == 0:  # coronal hole
                ax.contour(decoded_mask, colors='C0', linewidths=1)
            elif detect_result.label == 1:  # filament
                ax.contour(decoded_mask, colors='C2', linewidths=1)
        plt.colorbar(ax=ax)

        if current_id == 14:
            fig_id = int(total_id//15)+1
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'output-{fig_id}.{output_format}'))
            fig = plt.figure(figsize=(18, 24))
    # last figure
    fig_id = int(total_id//15)+1
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'output-{fig_id}.{output_format}'))

def plot_each(img_list, config_path="config.py", output_dir="vis", output_format='png'):
    # use agg backend to avoid memory leak
    matplotlib.use('agg')
    os.makedirs(output_dir, exist_ok=True)

    # Choose to use a config
    config = Config.fromfile(config_path)
    detection_model = init_detector(config=config_path,
                                    checkpoint=config['detection_checkpoint'],
                                    device=config['device'])

    for img in (pbar := tqdm(img_list)):
        out_img_name = img.split('/')[-1]
        pbar.set_description("current: " + out_img_name)
        detector = Detection(img, detection_model=detection_model, **(config['detection_config']))
        detector.detect()
        detector.plot(output=os.path.join(output_dir, out_img_name+'.'+output_format))

def main(img_list, config_path="config.py", output_dir="vis", plot_all_img=True, output_format='pdf'):
    if plot_all_img:
        plot_all(img_list=img_list, config_path=config_path, output_dir=output_dir, output_format=output_format)
    else:
        plot_each(img_list=img_list, config_path=config_path, output_dir=output_dir, output_format=output_format)

if __name__ == "__main__":
    img_list = sorted(glob.glob("/media/ExtHDD/data/CH_compare_dataset/193/*.fits"))
    main(img_list, config_path="checkpoints/DETACH_config_single.py", output_dir="vis_pdf", output_format='pdf')