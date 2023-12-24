import cv2
import numpy as np

img = cv2.imread('../data/cat2.png')
w, h, _ = img.shape

res = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

M_t = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img, M_t, (h, w))

M_r = cv2.getRotationMatrix2D((h / 2, w / 2), 45, 1)
rot = cv2.warpAffine(img, M_r, (h, w))

cv2.imshow('resized frame', res)
cv2.imshow('translated frame', dst)
cv2.imshow('rotated frame', rot)

k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()

for i in range(180):
    M_r = cv2.getRotationMatrix2D((h / 2, w / 2), i, 1)
    rot = cv2.warpAffine(img, M_r, (h, w))
    cv2.imshow('rotated frame', rot)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()


# affine transformation
# 一般需要借助标尺--标定板作为参考进行矫正
checker = cv2.imread("../data/checkerboard.png")
w, h, _ = checker.shape
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(checker, M, (h, w))
# 标定板上的直线没有改变

merge = np.hstack((checker, dst))
cv2.imshow('merged frame', merge)
cv2.waitKey()
cv2.destroyAllWindows()



#  Perspective Transformation
#  use chessboard and camera  example
'''
M = []
#M = cv2.getPerspectiveTransform(pts1, pts2)
checker_cor = cv2.warpPerspective(img, M, (300, 300))

merge = np.hstack((checker, checker_cor))
cv2.imshow('merged frame', merge)
cv2.waitKey()

cv2.destroyAllWindows()
'''