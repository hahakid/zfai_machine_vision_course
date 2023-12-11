import open3d as o3d
import numpy as np
import cv2

bag_filename = 'realsens.bag'

bag_reader = o3d.t.io.RSBagReader()
bag_reader.open(bag_filename)
im_rgbd = bag_reader.next_frame()


cv2.namedWindow("realsense", cv2.WINDOW_AUTOSIZE)

def process_frame(frame):
    rgb = np.asarray(frame.color)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = np.asarray(frame.depth)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.01), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = rgb.shape
    # resize based on depth image
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(rgb, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((rgb, depth_colormap))
    return images

while not bag_reader.is_eof():
    # process im_rgbd.depth and im_rgbd.color
    frame = process_frame(im_rgbd)
    cv2.imshow("realsense", frame)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
    im_rgbd = bag_reader.next_frame()

bag_reader.close()
cv2.destroyAllWindows()