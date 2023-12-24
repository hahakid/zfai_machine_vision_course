import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
尝试：
使用Morphological Gradient获取轮廓
再使用横向梯度进行处理
横向无变化=梯度小
纵向梯度变化陡峭=梯度大
'''
img = np.zeros((400, 400), np.uint8)
cv2.rectangle(img, (100, 50), (300, 350), 255, thickness=cv2.FILLED)
# lane example
#img = cv2.imread('../data/lane.png', 0)
#img = cv2.blur(img, ksize=(3, 3))

sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)

sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

merge = np.hstack((img, sobelx8u, sobel_8u))

cv2.imshow(" ", merge)
cv2.waitKey()
cv2.destroyAllWindows()