import cv2
import numpy as np
import random

# show all avaiable events
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON:  # change with different event
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(img, (x, y), random.randint(10, 50), color, -1)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()


