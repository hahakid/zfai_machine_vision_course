import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('../data/mrx.jpg')
# individual pixel modify
img[100, 100] = [255, 255, 255]
print(img[100, 100])

# individual pixel access
print('before:', img.item(10, 10, 2))

img.itemset((10, 10, 2), 100)
print('after:', img.item(10, 10, 2))


print('image shape:', img.shape)
print('image size (byte):', img.size)

w, h, c = img.shape

assert w * h * c == img.size

print("image data type:", img.dtype)

b, g, r = cv2.split(img)
img1 = cv2.merge((r, g, b))  # change channel order

#cv2.imshow(" ", img1)
#cv2.waitKey()

img2 = img.copy()
img2[:, :, 2] = 0  # set last channel 0
#cv2.imshow(" ", img2)
#cv2.waitKey()


BLUE = [255, 0, 0]

s_img = cv2.resize(img, (int(h/3), int(w/3)), cv2.INTER_NEAREST)

#cv2.imshow(" ", s_img)
#cv2.waitKey()

#

import matplotlib.patches as patches

h, w, _ = s_img.shape

#fig, ax = plt.subplots()
#im = ax.imshow(s_img)
#patch = patches.Circle((h/2, w/2), radius=h/2 if w > h else w/2, transform=ax.transData)
#im.set_clip_path(patch)
#plt.show()
#patch1 = np.asarray(im)


# change channels for matplotlib [rgb]
b, g, r = cv2.split(s_img)
s_img = cv2.merge((r, g, b))

replicate = cv2.copyMakeBorder(s_img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(s_img, 10, 10, 10, 10, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(s_img, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(s_img, 10, 10, 10, 10, cv2.BORDER_WRAP)
iso = cv2.copyMakeBorder(s_img, 10, 10, 10, 10, cv2.BORDER_ISOLATED)

# show different
cv2.imshow(" ", s_img)
cv2.waitKey()
#ax.axis("OFF")

plt.subplot(231)  # 2*3, first
plt.imshow(s_img)
plt.title('resized orange')
plt.axis("OFF")

plt.subplot(232)  # 2*3, 2nd
plt.imshow(replicate)
plt.title('replicate')
plt.axis("OFF")

plt.subplot(233)  # 2*3, 3rd
plt.imshow(reflect)
plt.title('reflect')
plt.axis("OFF")

plt.subplot(234)  # 2*3, 4th
plt.imshow(reflect101)
plt.title('reflect101')
plt.axis("OFF")

plt.subplot(235)  # 2*3, 5th
plt.imshow(wrap)
plt.title('wrap')
plt.axis("OFF")

plt.subplot(236)  # 2*3, 6th
plt.imshow(iso)
plt.title('iso')
plt.axis("OFF")


plt.show()


plt.close()
cv2.destroyAllWindows()
















