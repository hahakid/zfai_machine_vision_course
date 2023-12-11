import os
import cv2
import imageio
import numpy as np

path = '../data/magellan_telescope.mp4'

cap = cv2.VideoCapture(path)
count = 0
while cap.isOpened():
    print(count)
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1
    else:
        break
cap.release()
cv2.destroyAllWindows()
