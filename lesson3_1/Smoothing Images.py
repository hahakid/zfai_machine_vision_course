import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../data/mrx.jpg')
w, h, c = img.shape
'''
# kernel*kernel < denom: black
# kernel*kernel > denom: white
# kernel*kernel = denom: blur

k = 1 / denom * [ k*k ]
'''

k = 5  # odd
img = cv2.resize(img, (int(h/2), int(w/2)), cv2.INTER_CUBIC)

kernel = np.ones((k, k), np.float32) / (k * k)
dst_conv = cv2.filter2D(img, -1, kernel)
dst_blur = cv2.blur(img, (k, k))
dst_gBlur = cv2.GaussianBlur(img, (k, k), 0)
dst_mBlur = cv2.medianBlur(img, k)
dst_bf = cv2.bilateralFilter(img, 9, 75, 75)  #



for i in range(20):
    dst_conv = cv2.putText(dst_conv, 'Conv', (10, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
    dst_blur = cv2.putText(dst_blur, 'Blur', (10, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
    dst_gBlur = cv2.putText(dst_gBlur, 'Gauss', (10, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
    dst_mBlur = cv2.putText(dst_mBlur, 'Medium', (10, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
    dst_bf = cv2.putText(dst_bf, 'Bilateral', (10, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    merge = np.hstack((img, dst_conv, dst_blur, dst_gBlur, dst_mBlur, dst_bf))
    cv2.imshow(" ", merge)
    cv2.waitKey(100)
    dst_conv = cv2.filter2D(dst_conv, -1, kernel)
    dst_blur = cv2.blur(dst_blur, (k, k))
    dst_gBlur = cv2.GaussianBlur(dst_gBlur, (k, k), 0)
    dst_mBlur = cv2.medianBlur(dst_mBlur, k)
    dst_bf = cv2.bilateralFilter(dst_bf, 9, 75, 75)

cv2.waitKey()
cv2.destroyAllWindows()

