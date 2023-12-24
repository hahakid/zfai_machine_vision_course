import cv2
import numpy as np
from matplotlib import pyplot as plt
# https://www.cambridgeincolour.com/tutorials/histograms2.htm

img = cv2.imread('../data/hand.jpg', 0)

hist, bins = np.histogram(img.ravel(), 256, [0, 256])
cdf = hist.cumsum()
#fast_hist = np.bincount(img.ravel(), minlength=256) # 10 times faster
cdf_normalized = cdf * hist.max() / cdf.max()
plt.subplot(2, 1, 1)
plt.imshow(img)
plt.subplot(2, 1, 2)
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

img2 = cdf[img]
plt.subplot(2, 1, 1)
plt.imshow(img)
plt.subplot(2, 1, 2)
plt.plot(img2, color='b')

plt.hist(img2.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()


equ = cv2.equalizeHist(img)
res = np.hstack((img, equ)) #stacking images side-by-side
merge = np.vstack((img, equ))
cv2.imshow("", merge)
cv2.waitKey()
cv2.destroyAllWindows()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
merge = np.vstack((img, cl1))
cv2.imshow("", merge)
cv2.waitKey()
cv2.destroyAllWindows()
