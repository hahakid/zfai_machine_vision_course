# Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity.
import numpy as np
import cv2

lane = cv2.imread('../data/lane.png')
hand = cv2.imread("../data/hand.jpg")


def nothing(x):
    pass


modes = [cv2.RETR_EXTERNAL,  # retrieves only the extreme outer contours.
         cv2.RETR_LIST,  # retrieves all of the contours without establishing any hierarchical relationships.
         cv2.RETR_CCOMP,  # retrieves all of the contours and organizes them into a two-level hierarchy.
         cv2.RETR_TREE]  # retrieves all of the contours and reconstructs a full hierarchy of nested contours.
         #cv2.RETR_FLOODFILL]  # for 32-bit
methods = [cv2.CHAIN_APPROX_NONE,  # stores absolutely all the contour points.
           cv2.CHAIN_APPROX_SIMPLE,  # compresses horizontal, vertical, and diagonal segments and leaves only their end points.
           cv2.CHAIN_APPROX_TC89_L1,  # applies one of the flavors of the Teh-Chin chain approximation algorithm
           cv2.CHAIN_APPROX_TC89_KCOS]  # applies one of the flavors of the Teh-Chin chain approximation algorithm


def tuning_param(img):
    # set window
    cv2.namedWindow('image')
    # set bar
    cv2.createTrackbar('gray_thresh', 'image', 0, 255, nothing)
    cv2.createTrackbar('mode', 'image', 0, 3, nothing)
    cv2.createTrackbar('method', 'image', 1, 4, nothing)
    cv2.setTrackbarMin('method', 'image', 1)
    # default, modify based on tuning
    cv2.setTrackbarPos("gray_thresh", 'image', 141)
    cv2.setTrackbarPos("mode", 'image', 0)
    cv2.setTrackbarPos("method", 'image', 1)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:

        gray_thresh = cv2.getTrackbarPos('gray_thresh', 'image')
        mode = cv2.getTrackbarPos('mode', 'image')
        method = cv2.getTrackbarPos('method', 'image')
        # print(gray_thresh, mode, method)

        ret, thresh = cv2.threshold(imgray, gray_thresh, 255, 0)
        # @ input = an 8-bit single-channel image. @ contour retrieval mode @ contour approximation method
        contours, hierarchy = cv2.findContours(thresh, modes[mode], methods[method-1])

        with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 2)

        gray_stacked = cv2.merge((thresh, thresh, thresh))
        merge = np.vstack((gray_stacked, with_contours))
        cv2.imshow('image', merge)
        k = cv2.waitKey(5)

        if k == 27:
            break

    cv2.destroyAllWindows()


# tuning_param(lane)
tuning_param(hand)