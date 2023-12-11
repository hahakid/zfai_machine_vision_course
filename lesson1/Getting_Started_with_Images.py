import numpy as np
import cv2
# read image from @path @ cv2.IMREAD_COLOR | cv2.IMREAD_GRAYSCALE | cv2.IMREAD_UNCHANGED
img = cv2.imread('../data/cat2.png', 2)  # 0-gray, 1-color, 2-unchanged

# show image in window, wait and destroy window
cv2.imshow("window name1", img)
cv2.waitKey(0)  # 0-indefinitely time
#cv2.destroyWindow("window name")  # destroy specify window by name
cv2.destroyAllWindows()  # destroy all


# creat a window, and load image later
cv2.namedWindow("window name1", cv2.WINDOW_NORMAL)  # @cv2.WINDOW_AUTOSIZE
cv2.imshow("window name1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread()














