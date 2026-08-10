# XHAND 1 Pro 灵巧手 Python SDK 使用文档

## 一、Python 环境配置

1. 新建文件夹，将手相关 Python 文件、whl 包、SDK 包放入该文件夹

2. 首次安装依赖包（不加 sudo，在当前目录执行）

```Plain Text
pip install hand_controller-0.1.49-cp310-cp310-linux_x86_64.whl  # 按 Tab 补全实际包名
```

3. 强制重新安装（已安装时使用）

```Plain Text
pip install --force-reinstall ./hand_controller-0.1.49-cp310-cp310-linux_x86_64.whl
```

## 二、SDK 下载与基础配置

1. 安装 SDK 依赖包,版本号按tab自动补全

```Plain Text
sudo dpkg -i hand_sdk_0.1.49_ubuntu22.04_jammy_amd64.deb
```

2. 查看脚本帮助信息

```Plain Text
python3 xhand1pro_test.py -h
```

3. 查看 USB 设备名称

```Plain Text
ls /dev/ttyUSB*
```

4. 配置 USB 设备权限

```Plain Text
sudo chmod 777 /dev/ttyUSB0
or
sudo chmod 777 /dev/ttyACM0
```

5. ROS2 启动命令（测试需连接灵巧手）

```Plain Text
ros2 launch hand_control_ros2 hand_control.launch.py
```

## 三、Python 脚本命令

### 基础命令速查

```Plain Text
# 频率测试（阶跃测试-大角度）
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f hz_test  --hand-id 0
# 获取软件版本
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f get_version  --hand-id 0
# 获取位置数据
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f get_pos  --hand-id 0
# 获取序列号
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f get_sn  --hand-id 0
# 校准
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f calib  --hand-id 0
# 循环测试
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f cycle  --hand-id 0
# 步进测试
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f step  --hand-id 0
# 获取传感器数据（真实值）
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f get_sensor  --hand-id 0
# 获取手类型
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f hand_type  --hand-id 0
# 传感器清零（指定4号传感器）
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f sensor_clear -s 4  --hand-id 0 # 传感器清零（交互输入）
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f sensor_clear  --hand-id 0
```

### 全功能命令分类

#### 基础信息

```Plain Text
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f get_version --hand-id 0 # 获取固件版本
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f hand_type  --hand-id 0  # 获取手类型
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f get_sn    --hand-id 0   # 获取序列号
```

#### 标定与归零

```Plain Text
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f calib  --hand-id 0      # 关节标定
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f zero  --hand-id 0       # 所有关节安全回零
```

#### 状态读取

```Plain Text
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f get_pos  --hand-id 0    # 读取21个关节当前位置
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f get_sensor --hand-id 0  # 读取传感器数据
```

#### 运动测试

```Plain Text
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f step  --hand-id 0       # 阶跃测试
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f cycle  --hand-id 0      # 循环握拳/张开测试
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f play_pos --hand-id 0    # 交互式输入21个关节位置并执行
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f side_swing --hand-id 0  # 侧摆关节连续摆动测试（Ctrl+C停止）
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f clamp --hand-id 0       # 夹持测试
```

#### 传感器

```Plain Text
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f sensor_clear -s 1 --hand-id 0 # 清除指定传感器（-s传感器ID）
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f sensor_consistency --hand-id 0 # 传感器一致性测试
```

#### 固件升级

```Plain Text
# OTA升级（-b固件文件 -j关节ID）
python3 ./xhand1pro_test.py -d /dev/ttyUSB0  -f ota -b /home/user/Downloads/commboard_green_undf_undf_v1.1.1_20260209160606.bin -j 128  --hand-id 0 
```

## 四、传感器清零

```Plain Text
# 单个传感器清零（交互输入编号）
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f sensor_clear  --hand-id 0 
# 单个传感器清零（指定7号传感器）
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f sensor_clear -s 7  --hand-id 0 
# 遍历所有传感器清零
python3 xhand1pro_test.py -d /dev/ttyUSB0  -f sensor_clear_all  --hand-id 0 
```

## 五、关节 KP KD 修改

每个关节的 KP KD 均可配置修改，修改对应 JSON 文件即可，16 个关节可单独定制化参数。

## 六、EtherCAT 协议命令（需 sudo）

1. python环境配置

第一次配置：**ec必须加sudo\!\!\!**

```Plain Text
sudo pip install hand_controller-0.1.17-cp310-cp310-linux_x86_64.whl  # 按 Tab 补全实际包名
```

**如果已经安装过，需要强制重新安装：**

```Plain Text
**sudo** pip install --force-reinstall ./hand_controller-0.1.17-cp310-cp310-linux_x86_64.whl 
```



2. SDK下载：

```Plain Text
sudo dpkg -i hand_sdk_0.1.17_ubuntu22.04_jammy_amd64.deb
```

3. 查看脚本帮助信息

```Plain Text
**sudo** python3 xhand1pro_test.py -h
```

### 基础分类命令

enp3s0是网口名称，不同电脑网口不同，最好修改一下,改成enp3s0

```Plain Text
sudo ip link set oldname name newname
```

#### 基础信息

```Plain Text
sudo python3 xhand1pro_test.py -d enp3s0 -f hand_type --hand-id 0     # 获取手类型
sudo python3 xhand1pro_test.py -d enp3s0  -f get_sn    --hand-id 0    # 获取序列号
```

#### 标定与归零

```Plain Text
sudo python3 xhand1pro_test.py -d enp3s0  -f calib  --hand-id 0        # 关节标定
sudo python3 xhand1pro_test.py -d enp3s0  -f zero   --hand-id 0        # 所有关节安全回零
```

#### 状态读取

```Plain Text
sudo python3 xhand1pro_test.py -d enp3s0  -f get_pos  --hand-id 0      # 读取21个关节当前位置
sudo python3 xhand1pro_test.py -d enp3s0  -f get_sensor  --hand-id 0   # 读取传感器数据
```

#### 运动测试

```Plain Text
sudo python3 xhand1pro_test.py -d enp3s0  -f play_pos --hand-id 0 # 交互式输入21个关节位置并执行
sudo python3 xhand1pro_test.py -d enp3s0  -f side_swing --hand-id 0  # 侧摆关节连续摆动测试（Ctrl+C停止）
sudo python3 xhand1pro_test.py -d enp3s0  -f cycle --hand-id 0  # 循环握拳/张开测试
```

#### 传感器

```Plain Text
sudo python3 xhand1pro_test.py -d enp3s0  -f sensor_clear -s 1 --hand-id 0   # 清除指定传感器（-s传感器ID）
sudo python3 xhand1pro_test.py -d enp3s0 -t green -f sensor_consistency  # 传感器一致性测试
```

### 设备 ID 配置（通用）

```Plain Text
# 设置手的设备ID为30sudo python3 xhand1pro_test.py -d enp3s0 -f set_hand_id -j 30 --hand-id 0  
# 获取手的设备ID（单个ID）sudo python3 xhand1pro_test.py -d enp3s0 -f get_hand_id  
# 列出所有连接的手设备IDsudo python3 xhand1pro_test.py -d enp3s0 -f list_hand_ids --hand-id 0  
```

### OTA 升级通信板（EtherCAT，需 root）

```Plain Text
掌心版固件升级 传128
sudo python3 xhand1pro_test.py -d enp3s0  -f ota -b commboard_green_undf_undf_v1.1.5_20260323152529.bin -j 128 --hand-id 0  
通信版固件升级 传关节id
sudo python3 xhand1pro_test.py -d enp3s0  -f ota -b jointboard_green_undf_undf_v1.0.8_20260327011034.bin -j 0  --hand-id 0  
```



