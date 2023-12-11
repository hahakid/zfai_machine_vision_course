import numpy as np
import cv2
from matplotlib import pyplot as plt
# read image from @path @ cv2.IMREAD_COLOR | cv2.IMREAD_GRAYSCALE | cv2.IMREAD_UNCHANGED
img = cv2.imread('../data/cat2.png', 0)  # 0-gray, 1-color, 2-unchanged

# show image in window, wait and destroy window
cv2.imshow("window name1", img)
k = cv2.waitKey(0)  # 0-indefinitely time
if k == 27:  # assic 27 == ESC
    #cv2.destroyWindow("window name")  # destroy specify window by name
    cv2.destroyAllWindows()  # destroy all
elif k == ord('s'):  # s for save
    cv2.imwrite("../data/cat2_gray.png", img)
    cv2.destroyAllWindows()


plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([])  # clear x/y labels
plt.yticks([])
plt.show()






