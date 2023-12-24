# camshift: image segmentation
import cv2
import numpy as np
from matplotlib import pyplot as plt
# https://www.cambridgeincolour.com/tutorials/histograms2.htm

def nothing(x):
    pass
img = cv2.imread('../data/hand.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

target = hsv[50:450, 100:550]
cv2.namedWindow("image")
cv2.createTrackbar('thresh_h', 'image', 1, 255, nothing)  # 45 best
cv2.setTrackbarMin('thresh_h', 'image', 1)
cv2.createTrackbar('thresh_s', 'image', 1, 255, nothing)  # 135
cv2.setTrackbarMin('thresh_s', 'image', 1)
cv2.createTrackbar('thresh', 'image', 1, 255, nothing)  # 1
cv2.setTrackbarMin('thresh', 'image', 1)

while True:
    thresh_h = cv2.getTrackbarPos('thresh_h', 'image')
    thresh_s = cv2.getTrackbarPos('thresh_s', 'image')
    thresh_b = cv2.getTrackbarPos('thresh', 'image')

    roihist = cv2.calcHist([hsv], [0, 1], None, [thresh_h, thresh_s], [0, thresh_h, 0, thresh_s])

    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([target], [0, 1], roihist, [0, thresh_h, 0, thresh_s], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, thresh_b, 255, 0)
    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(target, thresh)
    res = np.hstack((target, thresh, res))
    cv2.imshow("image", res)
    k = cv2.waitKey(100)
    if k == 27:
        break

cv2.destroyAllWindows()









