import cv2
import numpy as np
img = cv2.imread('000001.png')

dets = np.array([[245.14649935, 244.46220333, 467.12020902, 755.22581547, 2],
        [45.14768727, 303.91372987, 218.63193766, 722.89196471, 1.]])
colours = np.random.rand(32, 3)
for d in dets:
    d = d.astype(np.int32)
    c = (0, 255, 0)
    cv2.rectangle(
        img, (d[0], d[1]), (d[2], d[3]),
        color=c, thickness=int(round(img.shape[0] / 256))
    )
    cv2.putText(img, f'{d[4]}', (d[0] - 9, d[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(img, f'{d[4]}', (d[0] - 8, d[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
cv2.imshow('result image', img)
cv2.waitKey(0)