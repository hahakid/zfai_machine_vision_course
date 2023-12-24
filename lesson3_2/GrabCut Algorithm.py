import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../data/hand.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = np.zeros(img.shape[:2], np.uint8)  # empty plane

bgdModel = np.zeros((1, 65), np.float64)  # Temporary array for the background model
fgdModel = np.zeros((1, 65), np.float64)  # Temporary arrays for the foreground model

rect = (100, 70, 500, 500)  # roi=[x1,y1, x2, y2]

# mask works as both input and output
# iter = 5
# mode = [GC_INIT_WITH_RECT = 0,
#     GC_INIT_WITH_MASK = 1,
#     GC_EVAL           = 2]
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]

plt.imshow(img)
#plt.colorbar()
plt.show()

# newmask is the mask image I manually labelled
newmask = cv2.imread('newmask.png', 0)

# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
# use previous mash from last step
mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

img = img * mask[:, :, np.newaxis]
plt.imshow(img)
#plt.colorbar(),
# plt.show()
