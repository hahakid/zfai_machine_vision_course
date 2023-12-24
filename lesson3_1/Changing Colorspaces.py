import cv2
import numpy as np

# prepare a blue object, or change to CNN model

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

cap = cv2.VideoCapture(0)
_, last_frame = cap.read()

while 1:
    _, frame = cap.read()
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lower_blue = np.array([130, 100, 100])  # change based on target
    upper_blue = np.array([170, 140, 140])
    # index, Checks if array elements lie between the elements of two other arrays.
    # channel1 ^ channel2 ^ channel3, channel之间也是与关系
    mask = cv2.inRange(rgb_img, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)  # 与计算， 仅保留掩码内像素。
    #  print(frame.shape, mask.shape, res.shape)
    merge = np.hstack([frame, res])
    last_frame = frame
    cv2.imshow('frame', merge)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()













