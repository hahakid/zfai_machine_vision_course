import cv2
img1 = cv2.imread("../data/realsense/lidar_camera_flower_color.jpg")
img2 = cv2.imread("../data/realsense/lidar_camera_flower_depth.jpg")
# dst = alpha * img1 + beta * img2 + gamma

for i in range(100):
    alpha = i / 100.0
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
    cv2.imshow('', dst)
    cv2.waitKey(20)
cv2.destroyAllWindows()

mrx = cv2.imread('../data/mrx.jpg')
logo = cv2.imread("../data/buaa.png")
w, h, c = logo.shape
roi = mrx[:w, :h]

logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)  # 反

cv2.imshow('', mask_inv)
cv2.waitKey()

mrx1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
mrx2_bg = cv2.bitwise_and(logo, logo, mask=mask)


dst = cv2.add(mrx1_bg, mrx2_bg)
mrx[:w, :h] = dst

cv2.imshow('', mrx)
cv2.waitKey()


cv2.destroyAllWindows()

