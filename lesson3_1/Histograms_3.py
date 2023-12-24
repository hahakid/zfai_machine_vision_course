import cv2
import numpy as np
from matplotlib import pyplot as plt
# https://www.cambridgeincolour.com/tutorials/histograms2.htm

img = cv2.imread('../data/hand.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# @input list, @ channel 0=h 1=s 3=v, @thresh 180=h 256=s,@ range
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

cdf = hist.cumsum()
#fast_hist = np.bincount(img.ravel(), minlength=256) # 10 times faster
cdf_normalized = cdf * hist.max() / cdf.max()


plt.subplot(3, 1, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # input
plt.subplot(3, 1, 2)
plt.plot(cdf_normalized, color='b')
#plt.plot(cdf, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.subplot(3, 1, 3)
plt.imshow(hist, interpolation='nearest')
plt.show()
