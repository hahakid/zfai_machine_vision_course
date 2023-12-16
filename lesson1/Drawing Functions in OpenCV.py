import numpy as np
import cv2
# parameters instructions
# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/group_imgproc_draw.html

# whiteboard @ width, height, channel, dtype
img = np.zeros((512, 512, 3), np.uint8)

#直线 @ source img, s_point, d_point, color, pixel-width
# color = [b,g,r]
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

# 矩形，# img, top-left-coord, down-right-coord, color, pixel-width
cv2.rectangle(img, (100, 0), (200, 100), (0, 255, 0), 4)

# 圆 @ center-coord, radius, color, pixel-width/-1=fill
cv2.circle(img, (400, 40), 50, (0, 0, 255), 2)

# 椭圆 @ center, (l-axe, s-axe), rotate angle, start-angle, end-angle, color, thickness
cv2.ellipse(img, (256, 256), (100, 50), 30, 0, 270, (255, 255, 2), -1)

#多边形 @ p1,p2,...,pn, dtype=np.int32 fixed usage
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
#print(pts)
#pts = pts.reshape((-1, 1, 2))
#print(pts)
# img, point list, isclosed, color, thickness, LineTypes
cv2.polylines(img, [pts], False, (0, 255, 255), 4, cv2.LINE_4)  # 空心

font = cv2.FONT_HERSHEY_SIMPLEX
# img, string, coord-down-left, font, font scale, color, thickness, line-type
cv2.putText(img, 'text', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)


cv2.imshow("window", img)
cv2.waitKey()
cv2.destroyAllWindows()