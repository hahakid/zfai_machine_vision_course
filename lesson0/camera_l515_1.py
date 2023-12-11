import numpy as np
import cv2
import pyrealsense2 as rs

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    print(s.get_info(rs.camera_info.name))

    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print("This demo requires Depth camera with color sensor")
    exit(0)
# @pixel= [1024*768, 1920*1080]
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # z16=depth, y8=intensity, c4=confidence

if device_product_line == 'L500':
    # L515, rgb = 1920*1080,6/15/30; 1280*720,6/15/30/60; 960*640,6/15/30/60, <--idea upper limitation
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    # other D400 series
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)  # use the configuration

try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()  # get frame
        depth_frame = frames.get_depth_frame()  # split depth from current frame
        color_frame = frames.get_color_frame()  # split color from current frame
        if not depth_frame and not color_frame:
            continue
        #depth_data = depth.as_frame().get_data()
        depth_image = np.asanyarray(depth_frame.get_data())  # to ndarray
        color_image = np.asanyarray(color_frame.get_data())  # to ndarray

        rgb_data = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # default opencv read BGR
        # use color mapping, depth,
        # depth_image[i] * alpha - beta
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.01), cv2.COLORMAP_JET)
        # get image size
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        # resize based on depth image
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow("realsense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("realsense", images)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break

except Exception as e:
    print(e)
    pass

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
