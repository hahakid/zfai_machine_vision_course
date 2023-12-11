# 硬件环境
1. realsense # 带支架或者其它安装载体，安装孔为英制1/4‑20 UNC 和 M4螺丝
2. Usb3.0 Type-C数据线
3. mini pc/jetson nano # 鼠标键盘显示器、网络和电源默认哈
# 操作系统(实测情况)
1. windows/ubuntu18 #建议双系统，不建议虚拟机。jetson仅支持ubuntu，但需要上位机刷机。
# 软件环境
1. anaconda # 虚拟环境管理
2. python  # 编程语言
3. pycharm community # 免费开源IDE环境
4. 清华源： https://mirrors.tuna.tsinghua.edu.cn/  # 用来提速包安装
    4.1 conda 源
    4.2 pypi 源
5. opencv tutorials https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_root.html
    5.1 pip install python-opencv
    5.2 相关函数还是要查看API手册
6. open3d https://github.com/isl-org/Open3D
    6.1 pip install open3d  # 目前版本是0.17，需要python3.8，ubuntu18及以上
7. realsense 官网及开发API  #希望ceo战略调整不要继续砍掉或者转卖 \cry
    7.1 https://www.intel.com/content/www/us/en/architecture-and-technology/realsense-overview.html
    7.2 https://dev.intelrealsense.com/docs/python2
    7.3 https://pypi.python.org/pypi/pyrealsense2

8. others，上述软件环境也可以自行在本地系统进行编译，环境变量配置和使用。