import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("../data/gradient.png", 0)

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # binary
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # inverse binary
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
ret, thresh6 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'TRIANGLE']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]

for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.axis("OFF")
plt.show()


#img = cv2.imread('../data/trafficlight2.png')
img = cv2.imread('../data/trafficlight.jpg')

# BGR  144 79 87
lower_red = np.array([0, 200, 0])
upper_red = np.array([255, 255, 255])

lower_green = np.array([200, 0, 0])
upper_green = np.array([255, 255, 255])

mask_red = cv2.inRange(img, lower_red, upper_red)
mask_green = cv2.inRange(img, lower_green, upper_green)

mask = np.bitwise_or(mask_red, mask_green)
res = cv2.bitwise_and(img, img, mask=mask)

merge = np.vstack((img, res))

cv2.imshow("", merge)
cv2.waitKey()
cv2.destroyAllWindows()



















