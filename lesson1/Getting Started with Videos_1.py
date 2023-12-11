import os
import cv2
import imageio

path = '../data/gif/'

img_list = sorted(os.listdir(path))
# print(img_list)
fps = 30
img = cv2.imread(os.path.join(path, img_list[0]))
img_size = img.shape  # h, w, c
print(img_size)

name = '../data/magellan_telescope.mp4'

'''# too huge for gif with inter-frame compression
name = './magellan.gif'

frames = []

for img in img_list:
    img = imageio.imread_v2(os.path.join(path, img))
    frames.append(img)

imageio.mimsave(name, frames, fps=10)
'''

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowriter = cv2.VideoWriter(name, fourcc, fps, (img_size[1], img_size[0]))
for img in img_list:
    image = cv2.imread(os.path.join(path, img))
    videowriter.write(image)

videowriter.release()



