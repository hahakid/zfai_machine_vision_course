# Upright-SURF robust upto [-15, 15] faster
# SURF orientation is calculated, slower

import cv2
import numpy as np

img = cv2.imread('../data/go.jpg', 0)
surf = cv2.xfeatures2d.SURF_create(400)  # patent
#surf.setHessianThreshold(50000)
#surf = cv2.xfeatures2d.SURF.create(hessianThreshold=400)  # Hessian threshold

kp, des = surf.detectAndCompute(img, None)

print(len(kp))

img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
cv2.imshow("", img2)
cv2.waitKey()
cv2.destroyAllWindows()




