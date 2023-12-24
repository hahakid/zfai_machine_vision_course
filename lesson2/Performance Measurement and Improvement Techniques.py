import cv2

img = cv2.imread('../../data/cat2.png')

def test_code(img):
    e1 = cv2.getTickCount()
    for i in range(5, 49, 2):
        img1 = cv2.medianBlur(img, i)
    e2 = cv2.getTickCount()
    t = (e2 - e1) / cv2.getTickFrequency()
    print(t)  # the smaller, the better performance of the computer


print("is optimized? ", cv2.useOptimized())
test_code(img)

cv2.setUseOptimized(False)
print("is optimized? ", cv2.useOptimized())
test_code(img)

