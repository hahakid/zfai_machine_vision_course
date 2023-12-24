# Contours properties
import numpy as np
import cv2

def nothing(x):
    pass

hand = cv2.imread("../data/sudoku.png")
hand_gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(hand_gray, 53, 255, 0)
# @ input = an 8-bit single-channel image. @ contour retrieval mode @ contour approximation method
contours, hierarchy = cv2.findContours(thresh, 1, 1)
cnt = contours[0]
M = cv2.moments(cnt)
print(M)

print('center: ', int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
print("area: ", cv2.contourArea(cnt))
print("arc length: ", cv2.arcLength(cnt, True))


def tuning_arc():
    cv2.namedWindow("image")
    cv2.createTrackbar('weight', 'image', 1, 10, nothing)
    while True:

        weight = cv2.getTrackbarPos('weight', 'image') / 10.0
        epsilon = weight * cv2.arcLength(cnt, True)
        #print(weight)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        #print(cnt)
        print(approx)
        with_contours = cv2.drawContours(hand, contours, -1, (255, 0, 0), 2)
        with_contours1 = cv2.drawContours(hand, approx, -1, (0, 255, 0), 2)
        merge = np.vstack((with_contours, with_contours1))
        cv2.imshow('image', merge)
        k = cv2.waitKey(10)
        if k == 27:
            break
    cv2.destroyAllWindows()

tuning_arc()
