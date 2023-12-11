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


profile = pipeline.start(config)  # use the configuration


depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)


clipping_distance_in_meters = 1  # distance threshold
clipping_distance = clipping_distance_in_meters / depth_scale


align_to = rs.stream.color
align = rs.align(align_to)  # auto resize to color image size
# use align for data

try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()  # get frame
        aligned_frames = align.process(frames)   # capable of getting both two data

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame and not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # to ndarray
        color_image = np.asanyarray(color_frame.get_data())  # to ndarray

        # threshold distance
        grey_color = 153  # color for background
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        # condition: remain [0, clipping_distance=1], 实际上0应该是没有矫正的，向前大概偏移了40cm， [0.4,1.4]
        # @ condition @ when Ture return, @ when False return
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        #rgb_data = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # default opencv read BGR
        # use color mapping, depth,
        # depth_image[i] * alpha - beta
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        # opencv imshow setting with quit button
        cv2.namedWindow("Align Example", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Align Example", images)
        k = cv2.waitKey(1)
        if k == 27 or k & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

except Exception as e:
    print(e)
    pass

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
