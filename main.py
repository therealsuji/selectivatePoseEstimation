import os
import argparse
import torch
from torch.utils.data import DataLoader
from multi_person_tracker import MPT
from utils.video_utils import (
    video_to_images, gen_masked_video, images_to_video)
import cv2
from tools.infer_video_d2 import generate2dKeypoints
from tools.prepare_data_2d_custom import process2D_keypoints
from tools.inference_3d import generate_3d_inference
yolo_img_size = 416
detector = "yolo"
tracker_batch_size = 3
display_box = False



def main(args):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    video_file = args.input_video
    if not os.path.isfile(video_file):
        exit(f'Input video \"{video_file}\" does not exist!')
    output_path = os.path.join(
        args.output_folder, os.path.basename(video_file).replace('.mp4', ''))
    os.makedirs(output_path, exist_ok=True)
    image_folder = video_to_images(video_file)

    mpt = MPT(
        device=device,
        batch_size=tracker_batch_size,
        display=display_box,
        detector_type=detector,
        output_format='list',
        yolo_img_size=yolo_img_size,
    )

    # result = mpt(image_folder)
    # gen_masked_video(image_folder, result, 2)
    # output_file_name = images_to_video(image_folder, 'masked_output.mp4')
    # keypoints_file = generate2dKeypoints(output_file_name)
    # process2D_keypoints(keypoints_file)
    generate_3d_inference('masked_output.mp4', True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_video', type=str,
                        help='input video path or youtube link')
    parser.add_argument('--output_folder', type=str,  default='output',
                        help='output folder to write results')
    args = parser.parse_args()

    main(args)
