# looking for specific patterns or specific features which are unique, can be easily tracked and can be easily compared.
# Harris Corner Detection
# corners are regions in the image with large variation in intensity in all the directions
# mathematical form： It basically finds the difference in intensity for a displacement of (u,v) in all directions.

import cv2
import numpy as np


def nothing(x):
    pass


# filename = '../data/checkerboard/online_save/000000.png'
filename = '../data/go.jpg'

cv2.namedWindow("image")
cv2.createTrackbar("bsize", 'image', 2, 10, nothing)
cv2.createTrackbar("ksize", 'image', 0, 10, nothing)
cv2.createTrackbar("k", 'image', 1, 100, nothing)

cv2.setTrackbarMin("bsize", 'image', 2)
cv2.setTrackbarMin("ksize", 'image', 0)  # odd && <31, for Sobel
cv2.setTrackbarMin("k", 'image', 4)

cv2.setTrackbarPos("bsize", 'image', 2)
cv2.setTrackbarPos("ksize", 'image', 0)
cv2.setTrackbarPos("k", 'image', 1)

img = cv2.imread(filename)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)  # int 2 float
# @input image @ block Size @ ksize @ k

flag = True #  False

while True:
    img1 = img.copy()
    bsize = cv2.getTrackbarPos("bsize", 'image')
    ksize = 3 + 2 * cv2.getTrackbarPos("ksize", 'image')
    k = cv2.getTrackbarPos("k", 'image') / 100
    print(bsize, ksize, k)
    dst = cv2.cornerHarris(gray, bsize, ksize, k)

    dst = cv2.dilate(dst, None)
    if flag:
        img1[dst > 0.01 * dst.max()] = [0, 0, 255]
    else:
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)  # float 2 int
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        '''
        termination criteria for iterative algorithms
        @ type: count, eps, count+eps
        @ maxCount: max # of iterations
        @ epsilon: desired accuracy, change in parameter
        '''
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        ''' 
        @ input image 
        @ winSize: Half of the side length of the search window. 
        @ zeroZone: Half of the size of the dead region in the middle of 
        the search zone over which the summation in the formula below is not done. 
        It is used sometimes to avoid possible singularities of the autocorrelation matrix. 
        The value of (-1,-1) indicates that there is no such a size.
        @ criteria: stops conditions
        '''
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
        res = np.hstack((centroids, corners))
        res = np.int0(res)

        img1[res[:, 1], res[:, 0]] = [0, 0, 255]  # Red
        img1[res[:, 3], res[:, 2]] = [0, 255, 0]  # Green

    # merge = np.vstack((gray, dst))
    cv2.imshow('image', img1)
    if cv2.waitKey(10) & 0xff == 27:
        break

cv2.destroyAllWindows()
