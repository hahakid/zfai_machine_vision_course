import cv2
import numpy as np
from matplotlib import pyplot as plt
# https://www.cambridgeincolour.com/tutorials/histograms2.htm
img = cv2.imread('../data/hand.jpg')
w, h, c = img.shape
color = ('r', 'g', 'b')

mask = np.zeros(img.shape[:2], np.uint8)
mask[int(w/4): int(w*3/4), int(h/4): int(h*3/4)] = 255
masked_img = cv2.bitwise_not(img, img, mask=mask)

for i, col in enumerate(color):
    # opencv
    # @ source mat list @ channels @ mask @ dims @ ranges
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])  # 40 times faster
    masked_histr = cv2.calcHist([img], [i], mask, [256], [0, 256])  # 40 times faster

    # numpy
    #histr, bins = np.histogram(img[:,:, i].ravel(), 256, [0, 256])
    fast_hist = np.bincount(img[:, :, i].ravel(), minlength=256) # 10 times faster
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

    plt.subplot(2, 2, 3)
    plt.imshow(masked_img)
    plt.subplot(2, 2, 4)
    plt.plot(masked_histr, color=col)
    plt.xlim([0, 256])

plt.show()



