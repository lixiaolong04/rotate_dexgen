# 本地 SDK 部署及注意事项

当前文档是在原生Ubuntu系统中使用，建议使用 Ubuntu 系统

# 部署流程

1. 安装 `docker`：运行以下命令进行安装，运行后，等待输出结束，输入 `8`（已经安装的话可以跳过）

```Bash
source <(wget -qO- http://fishros.com/install)
```

2. 设置 `docker`：配置环境，依次运行以下命令

```Shell
#添加docker用户组（一般此步可省略，docker会自动创建，） 
sudo groupadd docker 

#将用户加入到docker用户组，$USER为用户名。 
sudo usermod -aG docker $USER
sudo gpasswd -a $USER docker 

*#更新用户组
*newgrp docker

# 设置地址
sudo nano /etc/docker/daemon.json
# 输入以下字段
{
  "insecure-registries": ["10.128.114.9:5000", "reg.robotera.com:5000"]
}

# 重启 docker 
sudo systemctl restart docker

# 拉取镜像
# 拉取稳定版本镜像
docker pull 10.128.114.9:5000/ros-humble-nuc:latest
# 变换镜像名称
docker tag 10.128.114.9:5000/ros-humble-nuc:latest ros-humble-nuc:latest
```

3. 获取 `SDK` ：从 `jenkins` 上获取最新的压缩包：http://je\.robotera\.com:8080/job/robot\_sdk/

4. 创建目录，并解压缩文件至指定目录：

```Shell
sudo mkdir -p /opt/xbot /system_log/log/xbot_control

tar -xzf xbot_robot_sdk_dev_*.tar -C /opt/xbot
```

在 `SDK ≥ 2.0.110` 后，需要先部署 RL 算法，才能使用双足机器人的仿真环境

5. 创建 `docker` 容器：执行以下命令

- 本机无显卡

```Bash
docker run \
-d \
--name ros2_dev \
--privileged \
-v /opt/xbot:/root/xbot \
-v /opt/xbot/hardware/supervisor_config:/etc/supervisor/conf.d \
-v /system_log/:/system_log \
-v /dev:/dev \
-e AUDIO_GID=$(getent group audio | cut -d: -f3) \
-e PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native \
-v /run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse \
-v $HOME/.config/pulse/cookie:/root/.config/pulse/cookie \
--restart=always \
-e DISPLAY=$DISPLAY \
--network=host \
--cap-add=sys_nice \
ros-humble-nuc:latest
```

- 本机有显卡并已安装显卡驱动

```Bash
# 方法一
docker run \
-d \
--name ros2_dev \
--privileged \
-v /opt/xbot:/root/xbot \
-v /opt/xbot/hardware/supervisor_config:/etc/supervisor/conf.d \
-v /system_log/:/system_log \
-v /dev:/dev \
-e AUDIO_GID=$(getent group audio | cut -d: -f3) \
-e PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native \
-v /run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse \
-v $HOME/.config/pulse/cookie:/root/.config/pulse/cookie \
--restart=always \
-e DISPLAY=$DISPLAY \
--network=host \
--cap-add=sys_nice \
-d --security-opt seccomp=unconfined --gpus 'all,"capabilities=compute,utility,graphics"' \
ros-humble-nuc:latest

# 方法二（不卡顿）
docker run -it --name ros2_dev --network=host --security-opt seccomp=unconfined --gpus 'all,"capabilities=compute,utility,graphics"' -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --privileged=true -u root -v /dev:/dev -v /opt/xbot:/root/xbot 10.128.114.9:5000/ros-humble-nuc:latest /bin/bash
```

6. 创建进入 docker 容器的脚本

```Shell
echo -e '#!/bin/bash\ndocker exec -it ros2_dev /bin/bash' > go_to_docker.sh && chmod +x go_to_docker.sh
```

7. 进入容器修改 `ROS_DOMAIN_ID`为`1~255`的值，禁止使用`211`

```Bash
./go_to_docker.sh

export ROS_DOMAIN_ID=写非211值

grep -q "^export ROS_DOMAIN_ID=" ~/.bashrc \
&& sed -i "s/^export ROS_DOMAIN_ID=.*/export ROS_DOMAIN_ID=${NEW_DOMAIN_ID}/" ~/.bashrc \
|| echo "export ROS_DOMAIN_ID=${NEW_DOMAIN_ID}" >> ~/.bashrc

sed -i "s/ROS_DOMAIN_ID=\"[^\"]*\"/ROS_DOMAIN_ID=\"${NEW_DOMAIN_ID}\"/" \
/root/xbot/hardware/supervisor_config/supervisord.conf

source ~/.bashrc
```

8. 配置docker环境

```Bash
`#` 进入docker
./go_to_docker.sh

# 进入脚本目录
cd /root/xbot/hardware/install/scripts

# 运行环境部署脚本
./setup_env.sh

# 需改配置文件
vim /root/xbot/config.json
# 将 mode 值改为 sim ，修改 robot_type 为对应想仿真的机型：如 wr1, l3.4.3, m7
```

执行`plotjugger`或者运行`仿真环境`前，需要先在宿主机中运行`xhost +`

仿真环境中，必须设置`mode`为`sim`，`hardware_type`为`hand`

# **run docker 时，gpu报错处理方法**

1. **安装 NVIDIA Container Toolkit**

bash

复制

```Plain Text
# 添加仓库（Ubuntu/Debian）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

2. **配置 Docker 使用 NVIDIA 运行时**

bash

复制

```Plain Text
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

3. **验证安装是否成功**

bash

复制

```Plain Text
# 应该能看到 nvidia 运行时
docker info | grep -i nvidia

# 运行测试容器（会输出 GPU 信息）
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

如果测试容器能正常输出 `nvidia-smi` 信息，说明配置成功。

# config\.json 说明

|**参数名称**|**参数说明**|
|---|---|
|auto\_start|自启动设置<br>`true`表示开机自启动<br>`false`表示关闭开机自启动|
|enable\_developer|设置为`true`后，会取消自动配置机型和手类型，建议仿真环境设置为`true`<br>`true`表示自动配置整机机型及手类型<br>`false`表示不自动配置|
|l3\_mode|主要针对于 `l3.4` 真机机型使用，此值会在真机情况下覆盖 `mode` 参数值，参考值如下<br>`pd` 力控模式<br>`position` 位置模式<br>`velocity` 速度模式|
|mode|SDK 启动类型<br>`sim` 仿真模式<br>`pd` 力控模式<br>`position` 位置模式<br>`velocity` 速度模式|
|robot\_type|整机机型，仿真时，修改该值就可以切换成不同机型<br>`l3.4` L7 <br>`wr1` B4/Q5<br>`m7` M7 数采半身机器<br>`m7.1`凯龙\-固定立柱<br>`w7` 凯龙\-升降立柱<br>`l3.4.3` 马拉松机型|
|hardware\_type|是否有灵巧手，仿真必须为有手<br>`hand` 有手<br>`no_hand`无手|
|hand\_type|灵巧手类型<br>`xhand`：可用机型 l3\.4，wr1，m7，m7\.1，w7<br>`lite` ：可用机型 wr1，l3\.4\.3<br>`green` ：可用机型 l3\.4，m7|
|**以下参数一般不改，保持默认**||
|launch\_mode|xos启动模式|
|use\_mpc\_controller|mpc自启动|
|use\_rl\_controller|ri自启动|
|lift\_joint\_mode|立柱模式<br>`position` 位置模式<br>`velocity` 速度模式|

仿真环境的`config.json`示例（l3\.4 为例）

```JSON
{
    "auto_start": false,
    "enable_developer": true,
    "hand_type": "xhand",
    "hardware_type": "hand",
    "l3_mode": "pd",
    "launch_mode": "sim",
    "lift_joint_mode": "position",
    "mode": "sim",
    "robot_type": "l3.4",
    "use_mpc_controller": false,
    "use_rl_controller": false
}
```

