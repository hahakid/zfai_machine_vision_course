import cv2
import numpy as np

def nothing(x):
    pass

img = cv2.imread('../data/trafficlight.jpg', 0)
img = cv2.medianBlur(img, 5)


cv2.namedWindow("image")
cv2.createTrackbar("minDist", 'image', 1, 50, nothing)
cv2.createTrackbar("param1", 'image', 1, 50, nothing)
cv2.createTrackbar("param2", 'image', 1, 50, nothing)
cv2.setTrackbarMin("aperture", 'image', 1)
cv2.setTrackbarMax("aperture", 'image', 1)

while True:
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    min_Dist = cv2.getTrackbarPos("minDist", 'image')
    param1 = cv2.getTrackbarPos("param1", 'image')
    param2 = cv2.getTrackbarPos("param2", 'image')
    # @ only cv2.HOUGH_GRADIENT others for lines
    # @ Inverse ratio of the accumulator resolution to the image resolution.
    # @ Minimum distance between the centers of the detected circles
    # @ higher threshold of the two passed to the Canny edge detector
    # @ it is the accumulator threshold for the circle centers at the detection stage.
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=min_Dist, param1=param1, param2=param2, minRadius=10, maxRadius=50)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 255, 255), 3)

    cv2.imshow("image", cimg)
    k = cv2.waitKey(100)
    if k == 27:
        break

cv2.destroyAllWindows()