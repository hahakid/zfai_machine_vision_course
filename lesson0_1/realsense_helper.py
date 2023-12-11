import pyrealsense2 as rs

def get_profiles():
    ctx = rs.context()
    devices = ctx.query_devices()  # 设备列表

    color_profiles = []
    depth_profiles = []
    for device in devices:   # 遍历设备
        name = device.get_info(rs.camera_info.name)  # 获取 设备型号
        serial = device.get_info(rs.camera_info.serial_number)  # 获取 设备sn
        print('Sensor: {}, {}'.format(name, serial))
        print('Supported video formates:')
        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                stream_type = str(stream_profile.stream_type())

                if stream_type in ['stream.color', 'stream.depth']:
                    v_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()
                    video_type = stream_type.split(".")[-1]
                    print(' {}: width={}, height={}, fps={}, fmt={}'.format(video_type, w, h, fps, fmt))
                    '''
                    图像和深度均采用相同的流进行传输，因此，基于图w,h,以及每位的fmt确定总体截取长度，fps可能仅与配置有关，
                    实际实时速度应该是变化的
                    '''
                    if video_type == 'color':
                        color_profiles.append((w, h, fps, fmt))
                    else:
                        depth_profiles.append((w, h, fps, fmt))
    return color_profiles, depth_profiles


if __name__ == "__main__":
    c_p, d_p = get_profiles()
    print(c_p)
    print(d_p)