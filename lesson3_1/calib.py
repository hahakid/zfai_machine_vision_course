# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
import random

import cv2
import numpy as np
import cv2 as cv
import glob

# checkerboard: 6*8 grids, 5*7 internal corners, grid: 4cm*4cm and 8cm*8cm
# termination criteria
x = 5
y = 7
# @ desired accuracy @ maxcount @ type
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((y * x, 3), np.float32)
objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)
# print(objp)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


# images = ['../data/checkerboard.png', '../data/checkerboard2.png']

def check_corners(images, is_show=False):
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        # @img @ pattern size = (column, row)
        ret, corners = cv.findChessboardCorners(gray, (x, y), None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            if is_show:
                cv.drawChessboardCorners(img, (x, y), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey()

    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # print(ret, mtx, dist, rvecs, tvecs)
    if is_show:
        cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist


images = glob.glob('../data/checkerboard/online_save/' + "*.png")  # at least 10 frames, 4cm^2 8*6

mtx, dist = check_corners(images, is_show=True)

img = cv2.imread(images[random.randint(0, 19)])
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

merge = np.vstack((img, dst))
cv2.imshow("compare", merge)
cv2.waitKey()
cv2.destroyAllWindows()








