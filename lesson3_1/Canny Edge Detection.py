import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

def self_canny(img):
    img = cv2.blur(img, (5, 5))
    g_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    g_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    e_g = np.sqrt(g_x**2 + g_y**2)
    theta = np.arctan(g_y / (g_x + 0.0001))

    #fast = cv2.FastFeatureDetector_create()
    #kp = fast.detect(e_g, None)
    #filtered = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

    cv2.imshow("1", e_g)
    cv2.imshow("2", theta)
    cv2.waitKey()
'''
canny实际上包含一系列步骤：
1. 平滑降噪
2. 梯度图计算
3. 非极大值抑制 NMS，该操作在RPN的负样本抑制使用非常普遍，并有大量变种
4. Hysteresis Thresholding
'''
img = cv2.imread('../data/lane.png', 0)


cv2.namedWindow('image')
cv2.createTrackbar('t2', 'image', 0, 255, nothing)
cv2.createTrackbar('t1', 'image', 0, 255, nothing)

#local = self_canny(img)

edges = cv2.Canny(img, 0, 1)

while 1:
    merge = np.vstack((img, edges))
    cv2.imshow('image', merge)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    thresh1 = cv2.getTrackbarPos('t1', 'image')
    thresh2 = cv2.getTrackbarPos('t2', 'image')
    if thresh2 > thresh1:
        edges = cv2.Canny(img, thresh1, thresh2)
    else:
        edges = cv2.Canny(img, thresh1, thresh1+1)
        thresh2 = thresh1+1
        cv2.setTrackbarPos("t2", 'image', thresh2)



cv2.destroyAllWindows()