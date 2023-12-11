import numpy as np
import cv2
import pyrealsense2 as rs

cap = cv2.VideoCapture(1)  # 0 - default camera #

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # ESC/q for quit
            break
cap.release()
cv2.destroyAllWindows()



