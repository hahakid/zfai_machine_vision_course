# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
import os.path

import cv2
import numpy as np
import pyrealsense2 as rs

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()
count = 0
save_path = '../data/checkerboard/online_save/'

try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        rgb = frames.get_color_frame()  # video_frame
        rgb_data = rgb.as_frame().get_data()  # BufData
        rgb_image = np.asanyarray(rgb_data)  #ndarray
        gray_data = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # change grid pattern based on the checkerboard ()
        ret, corners = cv2.findChessboardCorners(gray_data, (5, 7), None)
        if ret:
            count += 1
            print(count, ret)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_path, str(count).zfill(6) + '.png'), rgb_image)
        gray_data = cv2.flip(gray_data, 1)
        cv2.imshow("rgb", gray_data)

        k = cv2.waitKey(300)  # have some differences

        if k == 27 or count > 20:
            # cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
