import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../data/trafficlight2.png')
img1 = cv2.imread('../data/trafficlight.jpg')

color_space = cv2.split(img)  # b, g, r
color_name = ["blue", "green", "red"]
for i, c in enumerate(color_space):
    ret2, th2 = cv2.threshold(c, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(c, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    merge = np.vstack((c, th2, th3))
    cv2.imshow(color_name[i], merge)
    cv2.waitKey()
cv2.destroyAllWindows()