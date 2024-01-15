# Shi-Tomasi Corner Detector
# R=λ_1 λ_2 − k * ( λ_1 + λ_2 )^2
# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/group_imgproc_feature.html#doxid-dd-d1a-group-imgproc-feature-1ga1d6bb77486c8f92d79c8793ad995d541
import cv2
import numpy as np


def nothing(x):
    pass


cv2.namedWindow("image")
cv2.createTrackbar("max", 'image', 10, 50, nothing)
cv2.createTrackbar("qlevel", 'image', 1, 40, nothing)
cv2.createTrackbar("minD", 'image', 1, 20, nothing)

cv2.setTrackbarMin("max", 'image', 10)
cv2.setTrackbarMin("qlevel", 'image', 1)  # odd && <31, for Sobel
cv2.setTrackbarMin("minD", 'image', 1)

cv2.setTrackbarPos("max", 'image', 10)
cv2.setTrackbarPos("qlevel", 'image', 1)
cv2.setTrackbarPos("minD", 'image', 10)

img = cv2.imread('../data/hand.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

while True:
    Mmax = cv2.getTrackbarPos("max", 'image')
    qlevel = cv2.getTrackbarPos("qlevel", 'image') / 100
    minD = cv2.getTrackbarPos("minD", 'image')

    img1 = img.copy()
    '''
    @ input: gray image
    @ maxCornets: maximum # of corners to return
    @ quality level: minimal eigenvalue 
    @ minDistance: distance between corners in pixel
    @ others refer website
    
    '''
    corners = cv2.goodFeaturesToTrack(gray, Mmax, qlevel, minD)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img1, (x, y), 3, 255, -1)

    cv2.imshow("image", img1)
    if cv2.waitKey(10) & 0xff == 27:
        break

cv2.destroyAllWindows()

















