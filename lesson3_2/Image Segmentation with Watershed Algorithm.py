import numpy as np
import cv2
from matplotlib import pyplot as plt


def nothing(x):
    pass

img = cv2.imread('../data/go.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.namedWindow("image")
cv2.createTrackbar("gray_min", 'image', 10, 255, nothing)
cv2.createTrackbar("gray_max", 'image', 100, 255, nothing)
cv2.createTrackbar("kernel", 'image', 0, 5, nothing)
cv2.createTrackbar("rate", 'image', 3, 10, nothing)
cv2.setTrackbarMin("rate", 'image', 3)

while True:

    gray_min = cv2.getTrackbarPos("gray_min", 'image')  # 0
    gray_max = cv2.getTrackbarPos("gray_max", 'image')  # 255
    k = cv2.getTrackbarPos("kernel", 'image') * 2 + 3
    rate = cv2.getTrackbarPos("rate", 'image') / 10.0
    if gray_min >= gray_max:
        cv2.setTrackbarPos("gray_max", 'image', gray_min + 1)
        cv2.setTrackbarPos("gray_min", 'image', gray_max - 1)

    ret, thresh = cv2.threshold(gray, gray_min, gray_max, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #ret, thresh = cv2.threshold(gray, gray_min, gray_max, cv2.THRESH_BINARY)

    kernel = np.ones((k, k), np.uint8)  # 2
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # distance_type
    dist_transform = cv2.distanceTransform(opening, distanceType=cv2.DIST_L2, maskSize=0)
    ret, sure_fg = cv2.threshold(dist_transform, rate * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    #img[markers == -1] = [255, 0, 0]
    gray[markers == -1] = 255
    #print(ret)
    merge = np.hstack((thresh, sure_fg, gray))
    cv2.imshow("image", merge)
    k = cv2.waitKey(100)
    if k == 27:
        break

cv2.destroyAllWindows()















