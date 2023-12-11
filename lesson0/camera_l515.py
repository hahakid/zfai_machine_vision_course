import numpy as np
import cv2
# First import the library
import pyrealsense2 as rs

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()

try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        rgb = frames.get_color_frame()  # video_frame

        depth_data = depth.as_frame().get_data()
        depth_image = np.asanyarray(depth_data)

        rgb_data = rgb.as_frame().get_data()  # BufData
        rgb_image = np.asanyarray(rgb_data)  #ndarray
        rgb_data = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        #print("depth:", type(depth), type(depth_data), type(depth_image))
        #print("image:", type(rgb), type(rgb_data), type(rgb_image))

        cv2.imshow("depth", depth_image)
        cv2.imshow("rgb", rgb_data)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
'''
cap = cv2.VideoCapture(1)  # 0 - default camera #

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # ESC/q for quit
            break
cap.release()
cv2.destroyAllWindows()
'''