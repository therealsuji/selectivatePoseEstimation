import os
import os.path as osp
import subprocess
import cv2
import glob
import numpy as np
import shutil

def video_to_images(vid_file,img_folder=None,):
    img_folder = 'output/tempfile/img'
    files = glob.glob(img_folder+"/*.png")
    for f in files:
        os.remove(f)
    os.makedirs(img_folder, exist_ok=True)
    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')
    # copies the first frame to separate folder for selection
    shutil.copy2('output/tempfile/img/000001.png', 'output/tempfile/first_frame/000001.png')
    return img_folder


def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]
    
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    return output_vid_file


def gen_masked_video(img_folder, trackers, track_id):
    
    image_file_names = sorted([
        osp.join(img_folder, x)
        for x in os.listdir(img_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    for frame, (img_fname, dets) in enumerate(zip(image_file_names, trackers)):
        image = cv2.imread(img_fname)
        for d in dets:
            d = d.astype(np.int32)
            track_id = int(track_id)
            if(track_id == d[4]):
                x = d[0]
                y = d[1]    
                w = d[2] - d[0]
                h = d[3] - d[1]

                x = makeValueOverZero(x)
                y = makeValueOverZero(y)
                w = makeValueOverZero(w)
                h = makeValueOverZero(h)

                mask = np.zeros(image.shape, np.uint8)
                mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
                cv2.imwrite(img_fname, mask)

def makeValueOverZero(val):
    if val <= 0:
        return 0
    return val    

 