import cv2
import numpy as np

img = cv2.imread('../data/lanting.png', 0)
_, img_b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#cv2.imshow(" ", img_b)
#cv2.waitKey()

def MT_test(kernel):
    # 侵蚀，消除了部分小噪点
    erosion = cv2.erode(img_b, kernel, iterations=1)
    # 膨胀，也增大了噪点
    dilation = cv2.dilate(img_b, kernel, iterations=1)

    # erosion followed by dilation.  # 暮右侧的噪点被消除，但影响了其他的字
    opening = cv2.morphologyEx(img_b, cv2.MORPH_OPEN, kernel)
    # Dilation followed by Erosion  #顺序相反但是，对噪点无影响
    closing = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel)

    #梯度 获取了轮廓，有点像局部反色
    gradient = cv2.morphologyEx(img_b, cv2.MORPH_GRADIENT, kernel)
    # 顶帽变换，the difference between input image and Opening of the image.
    tophat = cv2.morphologyEx(img_b, cv2.MORPH_TOPHAT, kernel)
    # 黑帽变换， the difference between the closing of the input image and input image
    blackhat = cv2.morphologyEx(img_b, cv2.MORPH_BLACKHAT, kernel)


    merge1 = np.hstack((img_b, erosion, dilation, opening))
    merge2 = np.hstack((closing, gradient, tophat, blackhat))
    merge = np.vstack((merge1, merge2))

    cv2.imshow(" ", merge)
    cv2.waitKey()
    cv2.destroyAllWindows()


'''
input: binary images
basic operation: Erosion and Dilation
'''
#kernel = np.ones((3, 3), np.uint8)  # equal to square_kernel
k_size = 7
#self_defined
# an application: definition of the dilated convolutional kernel, get larger reception field without enlarge parameters
square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
ellip_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
circle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

kernels = [square_kernel, ellip_kernel, circle_kernel]

for k in kernels:
    print(k)
    MT_test(k)










