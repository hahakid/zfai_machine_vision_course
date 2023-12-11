#http://www.open3d.org/docs/latest/index.html

import open3d as o3d
import numpy as np
import cv2
import json


'''
[Open3D INFO] [0] Intel RealSense L515: f0221803
[Open3D INFO] 	color_format: 
[RS2_FORMAT_BGR8 | RS2_FORMAT_BGRA8 | RS2_FORMAT_RGB8 | RS2_FORMAT_RGBA8 | RS2_FORMAT_Y16 | RS2_FORMAT_YUYV]
[Open3D INFO] 	color_resolution: [1280,720 | 1920,1080 | 640,360 | 640,480 | 960,540]
[Open3D INFO] 	color_fps: [15 | 30 | 6 | 60]
[Open3D INFO] 	depth_format: [RS2_FORMAT_Z16]
[Open3D INFO] 	depth_resolution: [1024,768 | 320,240 | 640,480]
[Open3D INFO] 	depth_fps: [30]
[Open3D INFO] 	visual_preset: 
[RS2_L500_VISUAL_PRESET_CUSTOM | RS2_L500_VISUAL_PRESET_DEFAULT | 
RS2_L500_VISUAL_PRESET_LOW_AMBIENT | RS2_L500_VISUAL_PRESET_MAX_RANGE | 
RS2_L500_VISUAL_PRESET_NO_AMBIENT | RS2_L500_VISUAL_PRESET_SHORT_RANGE]
[Open3D INFO] Open3D only supports synchronized color and depth capture (color_fps = depth_fps).
'''
config_filename = './l515.json'  # need a full configuration setting
bag_filename = './realsens.bag'
ifSave = False

with open(config_filename) as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))
o3d.t.io.RealSenseSensor.list_devices()

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

rs = o3d.t.io.RealSenseSensor()

if ifSave:
    rs.init_sensor(sensor_config=rs_cfg, sensor_index=0, filename=bag_filename)
else:
    rs.init_sensor(rs_cfg, 0)  # without save bag, save disk

rs.start_capture(True)

cv2.namedWindow("realsense", cv2.WINDOW_AUTOSIZE)

for fid in range(50):
    im_rgbd = rs.capture_frame(True, True)
    rs.pause_record()
    frame = process_frame(im_rgbd)
    cv2.imshow("realsense", frame)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
    rs.resume_record()

rs.stop_capture()
cv2.destroyAllWindows()





