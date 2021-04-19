import torch
from multi_person_tracker import MPT
import cv2
import numpy as np

yolo_img_size = 416
detector = "maskrcnn"
tracker_batch_size = 3
display_box = False


#MPT -> sort.py -> 184 ->       KalmanBoxTracker.count = 0

class Tracking:

    def init(self):
        self.mpt = MPT(
            device=self.device,
            batch_size=tracker_batch_size,
            display=display_box,
            detector_type=detector,
            output_format='list',
            yolo_img_size=yolo_img_size,
        )

    def __init__(self):
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.init()

    def track_complete_file(self):        
        result = self.mpt('output/tempfile/img')
        return result

    def track_first_frame(self):
        result = self.mpt('output/tempfile/first_frame')
        return result

    def get_tracked_image(self, trackers):
        img = cv2.imread('output/tempfile/first_frame/000001.png')
        colours = np.random.rand(32, 3)
        for d in trackers:
            d = d.astype(np.int32)
            c = (0, 255, 0)
            cv2.rectangle(
                img, (d[0], d[1]), (d[2], d[3]),
                color=c, thickness=int(round(img.shape[0] / 256))
            )
            cv2.putText(img, f'{d[4]}', (d[0] - 9, d[1] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(img, f'{d[4]}', (d[0] - 8, d[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        cv2.imwrite('output/tempfile/first_frame/000001.png', img)
        return 'output/tempfile/first_frame/000001.png'
