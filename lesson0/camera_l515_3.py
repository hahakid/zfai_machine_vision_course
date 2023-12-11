import numpy as np
import cv2
import pyrealsense2 as rs
import argparse
import os


parser = argparse.ArgumentParser(description='Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.')
parser.add_argument('-i', '--input', type=str, help='bag path')
args = parser.parse_args()

if not args.input:
    print("No input parameters.")
    exit()

if os.path.splitext(args.input)[1] != '.bag':
    print('Error data format.')
    exit()

try:

    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, args.input)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    profile = pipeline.start(config)  # use the configuration

    cv2.namedWindow("Bag steam", cv2.WINDOW_AUTOSIZE)

    colorizer = rs.colorizer()
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        cv2.imshow("Bag steam", depth_color_image)
        k = cv2.waitKey(1)
        if k == 27 or k & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

finally:
    pass
    cv2.destroyAllWindows()


# python .\lesson0\camera_l515_3.py -i .\data\test.bag