# Scale-Invariant Feature Transform  尺度/放缩不变性

import cv2
import numpy as np
print(cv2.__version__)
img = cv2.imread("../data/hand.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()  # be careful of the function name in new version
# @ input @ mask
kp = sift.detect(gray, None)

#img = cv2.drawKeypoints(gray, kp, img)  # show simple
img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # show detail
# cv2.imwrite('sift_keypoints.jpg', img)
cv2.imshow("", img)
cv2.waitKey()
cv2.destroyAllWindows()

















