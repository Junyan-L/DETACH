import os
import cv2


def save_video(save_root, output_path):
    buffer_img_list = sorted(os.listdir(save_root))
    image_ori = cv2.imread(os.path.join(save_root, buffer_img_list[0]))

    video_size = (image_ori.shape[1], image_ori.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 10, video_size, True)

    for img in buffer_img_list:
        frame = cv2.imread(os.path.join(save_root, img))
        video.write(frame)
    video.release()


if __name__ == "__main__":
    save_video("output", "video/2010-05+.mp4")