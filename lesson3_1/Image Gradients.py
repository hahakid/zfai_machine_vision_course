import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("../data/sudoku.png", 0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
for i in range(255):
    print(i)
    ret, img_b = cv2.threshold(img, i, 255, cv2.THRESH_BINARY)
    cv2.imshow(" ", img_b)
    cv2.waitKey()
'''

ret, img_b = cv2.threshold(img, 48, 255, cv2.THRESH_BINARY)

laplacian = cv2.Laplacian(img_b, cv2.CV_64F)

for i in range(1, 13, 2):
    print(i)

    sobelx = cv2.Sobel(img_b, cv2.CV_64F, 1, 0, ksize=i)

    sobely = cv2.Sobel(img_b, cv2.CV_64F, 0, 1, ksize=i)

    merge = np.vstack((np.hstack((img_b, laplacian)), np.hstack((sobelx, sobely))))

    cv2.imshow(" ", merge)
    cv2.waitKey()

cv2.destroyAllWindows()