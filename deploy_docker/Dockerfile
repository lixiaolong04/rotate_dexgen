# Xhand sim2real 真机部署容器 —— Ubuntu 22.04 (jammy), 对齐厂商 SDK 编译目标
#
# 为什么用 22.04: hand_sdk .deb + hand_controller wheel 都是给 jammy 编的,
#   依赖 libfmt.so.8 / libssl.so.3 等 22.04 的 soname; 在 24.04 上装不上(libfmt 是 9)。
#   容器里用原生 22.04 => .deb 直接装得上、wheel 开箱即用, 不用跟主系统较劲。
#
# 镜像里烤进: 系统 SDK(.deb) + Python 绑定(hand_controller) + CPU 版 torch 等推理依赖。
# 项目代码 + 当前部署权重 + cache 在【运行时挂载】到 /workspace (改代码不用重建镜像)。
#
# 构建上下文由 build.sh 准备(把带空格/(1) 的 SDK 文件规整成干净名字):
#   ./hand_sdk.deb        <- xhandSDK/hand_sdk_*.deb
#   ./hand_controller/    <- xhandSDK/hand_controller-*/hand_controller (解包后的包目录)
FROM xhand-deploy:offline

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV XHAND_DEPLOY_OFFLINE_ONLY=0

# Xhand C++ SDK 系统包: 用 apt 装本地 .deb, 自动解析 libfmt8/libssl3/glog/boost 等依赖
COPY hand_sdk.deb /tmp/hand_sdk.deb
RUN apt-get update && apt-get install -y /tmp/hand_sdk.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/hand_sdk.deb

# Xhand Python 绑定(解包后的 wheel 包 -> dist-packages); 让 loader 找到包内自带的 libhand_sdk.so
COPY hand_controller /usr/local/lib/python3.10/dist-packages/hand_controller
RUN echo "/usr/local/lib/python3.10/dist-packages/hand_controller" \
        > /etc/ld.so.conf.d/hand_controller.conf \
    && ldconfig

# CPU torch/numpy/termcolor/tensorboardX/pytest are inherited from the
# already-validated xhand-deploy:offline base image.

# 键盘状态机(e/w/q/t)用 pynput; 借鉴 SharpaWave 的 utils/Dockerfile: 装 evdev 编译依赖后再 pip。
# (容器内抓键还需 X11 转发, 见 docker-compose.yml 的 /tmp/.X11-unix + DISPLAY; 或用 --no_keyboard。)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-dev libevdev-dev \
    && pip3 install --no-cache-dir pynput \
    && apt-get remove -y build-essential python3-dev libevdev-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# 官方 SDK 的扫频/绘图工具需要 matplotlib。使用 Jammy 官方 APT 包，
# 避免外部 PyPI 镜像偶发的 wheel 哈希不一致。
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# Jammy 的 Matplotlib 扩展使用 NumPy 1.x ABI；移除基础镜像中的 pip NumPy
# 2.x，使 Python 使用随 Matplotlib 安装的系统 NumPy 1.x。
RUN pip3 uninstall -y numpy

# 项目运行时挂载到 /workspace; 设 PYTHONPATH 让 `python -m ...` 和 `python scripts/...` 都能 import
ENV PYTHONPATH=/workspace
WORKDIR /workspace

CMD ["bash"]
