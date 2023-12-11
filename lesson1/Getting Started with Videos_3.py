import os
import cv2
import imageio
import numpy as np

path = '../data/magellan_telescope.mp4'
out_path = '../data/magellan_telescope_flip.mp4'
cap = cv2.VideoCapture(path)

count = 0

ret, frame = cap.read()  # frame 0

if ret:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
    while cap.isOpened():
        print(count)
        if ret == True:
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.flip(frame, 0)  # 0-horizontal 1-vertical 2-both
            resize = cv2.resize(frame, (int(frame.shape[1]/3), int(frame.shape[0]/3)))
            cv2.imshow("frame", resize)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                out.write(frame)
                break
        else:
            break
        ret, frame = cap.read()
        count += 1
else:
    print("empty video frame, check video file.")

cap.release()
cv2.destroyAllWindows()
