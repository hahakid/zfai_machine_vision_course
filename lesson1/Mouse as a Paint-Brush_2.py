import cv2
import numpy as np
import random

drawing = False
mode = True
ix, iy = -1, -1

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    if event == cv2.EVENT_FLAG_LBUTTON:  # change with different event
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), color, -1)
            else:
                cv2.circle(img, (x, y), random.randint(0, 50), color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), color, -1)
        else:
            cv2.circle(img, (x, y), random.randint(0, 50), color, -1)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()


