# Hough Transform to detect any shape

import cv2
import numpy as np

def nothing(x):
    pass

img = cv2.imread('../data/sudoku.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.namedWindow("image")
cv2.createTrackbar("gray_min", 'image', 0, 255, nothing)
cv2.createTrackbar("gray_max", 'image', 0, 255, nothing)
cv2.createTrackbar("aperture", 'image', 0, 2, nothing)
cv2.setTrackbarMin("aperture", 'image', 0)
cv2.setTrackbarMax("aperture", 'image', 2)
cv2.createTrackbar("rho", 'image', 1, 10, nothing)
cv2.setTrackbarMin("rho", 'image', 1)
cv2.createTrackbar("theta", 'image', 1, 180, nothing)
cv2.setTrackbarMin("theta", 'image', 1)
cv2.createTrackbar("accum", 'image', 50, 250, nothing)
cv2.setTrackbarMin("accum", 'image', 50)
cv2.createTrackbar("llength", 'image', 10, 200, nothing)
cv2.createTrackbar("lgap", 'image', 1, 10, nothing)



while True:
    gray_min = cv2.getTrackbarPos("gray_min", 'image')
    gray_max = cv2.getTrackbarPos("gray_max", 'image')
    if gray_min >= gray_max:
        cv2.setTrackbarPos("gray_max", 'image', gray_min + 1)
    aperture = 3 + cv2.getTrackbarPos("aperture", 'image') * 2
    theta_t = cv2.getTrackbarPos("theta", 'image')
    rho_t = cv2.getTrackbarPos("rho", 'image')
    accum = cv2.getTrackbarPos("accum", 'image')
    llength = cv2.getTrackbarPos("llength", 'image')
    lgap = cv2.getTrackbarPos("lgap", 'image')

    edges = cv2.Canny(gray, gray_min, gray_max, apertureSize=aperture)
    # with two additional params @ minlineLength, maxLineGap
    lines = cv2.HoughLinesP(edges, rho_t, np.pi/180 * theta_t, accum, llength, lgap)
    img1 = img.copy()

    for line in lines:
        #rho, theta = line[0]
        x1, y1, x2, y2 = line[0]
        cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)

    merge = np.hstack((cv2.merge((edges, edges, edges)), img1))
    cv2.imshow("image", merge)

    k = cv2.waitKey(100)
    if k == 27:
        break

cv2.destroyAllWindows()












