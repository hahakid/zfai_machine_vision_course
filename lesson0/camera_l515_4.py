import numpy as np
import cv2
import pyrealsense2 as rs
# import argparse
# import os
import math
import time
import open3d as o3d

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# for realsense
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    # print(s.get_info(rs.camera_info.name))
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print('need a camera, but cannot find')
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

pipeline.start(config)

profile = pipeline.get_active_profile()
# aaa = profile.get_steams(rs.stream.depth)
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))

depth_intrinsics = depth_profile.get_intrinsics()  # 内参
w, h = depth_intrinsics.width, depth_intrinsics.height

pc = rs.pointcloud()
decimate = rs.decimation_filter()  # 中值过滤
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()


def mouse_cb(event, x, y, flags, param):
    # 鼠标左键
    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    # 鼠标右键
    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    # 鼠标中键
    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    # 鼠标移动
    if event == cv2.EVENT_MOUSEMOVE:
        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]
        # 左键－旋转视角更新
        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2
        # 右键－平移
        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        # 中键放缩
        elif state.mouse_btns[2]:
            dz = math.sqrt(dx ** 2 + dy ** 2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


# 窗口定义
cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)  # init－自适应窗口
cv2.resizeWindow(state.WIN_NAME, w, h)  # 基于ｗｈ动态调节
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)  # 鼠标action使能


def project(v):
    h, w = out.shape[:2]
    view_aspect = float(h) / w
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * (w * view_aspect, h) + (w / 2.0, h / 2.0)

    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n + 1):
        x = -s2 + i * s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)), view(pos + np.dot((x, 0, s2), rotation)), color)

    for i in range(0, n + 1):
        z = -s2 + i * s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)), view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    line3d(out, pos, pos + np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos + np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos + np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    # camera
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        '''
        (0,0)--------(w,0)
        |               |
        |               | 
        (0,h)--------(w,h)
        '''
        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_left = get_point(0, h)
        bottom_right = get_point(w, h)
        '''
                line1-->     
               
        line4 ^            line2 |
                
                <--line3
        '''
        line3d(out, view(top_left), view(top_right), color)  # line1
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    if painter:
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5 ** state.decimate

    h, w = out.shape[:2]
    j, i = proj.astype(np.uint32).T
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T

    np.clip(u, 0, ch - 1, out=u)
    np.clip(v, 0, cw - 1, out=v)

    out[i[m], j[m]] = color[u[m], v[m]]

def pc_show(pc, norm_flag=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800)
    opt = vis.get_render_option()
    opt.point_size = 2
    opt.point_show_normal = norm_flag
    for p in pc:
        vis.add_geometry(p)
    vis.run()
    vis.destroy_window()

def save_pcd(color, depth):
    assert color.shape == depth.shape
    if color.shape[2] == 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth)
        pcd.colors = o3d.utility.Vector3dVector(color)
        pc_show(pcd)


out = np.empty((h, w, 3), dtype=np.uint8)

while True:
    if not state.paused:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)

    now = time.time()

    out.fill(0)

    grid(out, (0, 0.5, 1), size=1, n=10)
    frustum(out, depth_intrinsics)
    axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, color_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, color_source)
        tmp = cv2.resize(tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s " %
                       (w, h, 1.0 / dt, dt * 1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord('r'):
        state.reset()

    # single pressed / double pressed
    if key == ord('p'):
        state.paused ^= True
    # down sampling
    if key == ord('d'):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    #
    if key == ord('z'):
        state.scale ^= True

    if key == ord('c'):
        state.color ^= True

    if key == ord('s'):
        cv2.imwrite("./out.png", out)

    if key == ord('e'):
        # only point cloud format in this wrapper, ply
        # can be translated into .pcd based Open3d
        # Bug: in JET colormap mode, save pc rises a
        # print(type(mapped_frame), mapped_frame) #@2 = pyrealsense2.frame BGR8 (can be saved)
        #   @2 = pyrealsense2.frame Z16 (cannot be saved)
        # TODO turn into open3d.PCD formate
        points.export_to_ply('./out.ply', mapped_frame)

    if key in (27, ord('q')) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

pipeline.stop()
