import cv2
import numpy as np

# feature Pyramids 是深度学习特征构建的常规操作，
# 常用于适配因透视坐标系导致的目标成像规模变化
#
shape = (512, 512)  #  2^8 = 512
A = cv2.imread('../data/xukun.png')
A = cv2.resize(A, shape, interpolation=cv2.INTER_CUBIC)
B = cv2.imread('../data/dog.png')
B = cv2.resize(B, shape, interpolation=cv2.INTER_CUBIC)
#assert A.shape == B.shape

G = A.copy()
gpA = [G]
for i in range(6):
    #print(i, G.shape)
    G = cv2.pyrDown(G)
    gpA.append(G)

G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    aa = gpA[i-1]
    bb = GE
    L = cv2.subtract(gpA[i-1], GE)
    lpA.append(L)

lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1], GE)
    lpB.append(L)


LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:int(cols/2)], lb[:, int(cols/2):]))
    #ls = np.hstack((la[:, 0:cols], lb[:, cols:]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

real = np.hstack((A[:, :int(cols/2)], B[:, int(cols/2):]))
#real = np.hstack((A, B))

merge = np.hstack((ls_, real))
cv2.imshow('merged frame', merge)
cv2.waitKey()
cv2.destroyAllWindows()
















