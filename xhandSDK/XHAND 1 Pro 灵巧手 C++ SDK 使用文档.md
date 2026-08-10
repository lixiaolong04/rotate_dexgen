# XHAND 1 Pro 灵巧手 C\+\+ SDK 使用文档

## 环境配置

SDK 需运行在 **Ubuntu 系统环境** 下，且需安装依赖包：

```Plain Text
sudo apt install libspdlog-dev libboost-all-dev libfmt-dev libssl-dev
sudo dpkg -i hand_sdk_0.1.49_ubuntu22.04_jammy_amd64.deb
pip3 install hand_controller-0.1.49-cp310-cp310-linux_x86_64.whl
```

---

## 使用示例

### ① 串口（Serial）示例

```Plain Text
// 串口auto serial_comm = std::make_shared<SerialCommunication>("/dev/ttyUSB0", 3000000, 1000);if(serial_comm->open() != ErrorCode::SUCCESS) {LOG_ERROR("Failed to open serial communication");}

GREEN_HAND::GreenHand hand(0, serial_comm);
std::string version;auto code = hand.get_software_version(version);LOG_INFO("Version:{} code: {}", version, (int)code);auto code = hand.set_joint_cmd(index, GREEN_HAND::JointCommand_User{100, 0, 0, deg, 350, 3, 0, 0 , 0, 0});auto joint_data = hand.get_joint_state(index);float current_pos = *(float32 *)&joint_data.position;LOG_INFO("write pos: {} , current pos: {}", deg, current_pos);
```

### ② EtherCAT 示例

```Plain Text
// Ethercatauto ec_comm = std::make_shared<EtherCatCommunication>("eth0");if(ec_comm->open() != ErrorCode::SUCCESS) {LOG_ERROR("Failed to open ec communication");}

GREEN_HAND::GreenHand hand(0, ec_comm);
std::string version;auto code = hand.get_software_version(version);LOG_INFO("Version:{} code: {}", version, (int)code);auto code = hand.set_joint_cmd(index, GREEN_HAND::JointCommand_User{100, 0, 0, deg, 350, 3, 0, 0 , 0, 0});auto joint_data = hand.get_joint_state(index);float current_pos = *(float32 *)&joint_data.position;LOG_INFO("write pos: {} , current pos: {}", deg, current_pos);
```

### 



---

# XHAND 1 Pro API 参考

## SerialCommunication 串口通信接口文档

### 接口功能

BaseHand 提供对 XHAND Pro 手设备的统一抽象：发送 / 接收帧、寄存器读写、关节命令与状态读写、传感器读写、标定与设备枚举等。支持不同底层通信（串口 / CAN / EtherCAT）。

- 串口 / CAN：基于帧通信 

- EtherCAT：基于 SDO 通信，并维护 device\_id 与 slave\_id 映射

### 构造函数

- **接口功能**：创建 BaseHand 对象并绑定通信实现，若 comm 有效则自动初始化设备映射（EtherCAT 下建立 device\_id → slave\_id 映射）

- **接口声明**：`explicit BaseHand(uint8_t device_id, std::shared_ptr<CommunicationBase> comm);`

注意事项：

comm 不能为空，否则会抛出异常 

构造函数内部会调用 init\_device\_slaves\(\) 

device\_id 为逻辑设备 ID（0\~255）

- **参数说明**：无

- **返回值**：无

- **注意**：对象创建后不会立即打开串口，需要调用 `open()`

### open

- **接口功能**：打开串口并绑定唯一的 device\_id

- **接口声明**：`ErrorCode open() override;`

- **参数说明**：无

- **返回值**：无

### close

- **接口功能**：关闭当前串口连接

- **接口声明**：`ErrorCode close() override;`

- **参数说明**：无

- **返回值**：无

### send

- **接口功能**：发送原始二进制数据到串口

- **接口声明**：`ErrorCode send(uint8_t , const uint8_t *data, int len);`

- **参数说明**：无

- **返回值**：无

### receive

- **接口功能**：从串口接收原始数据

- **接口声明**：`size_t receive(uint8_t , uint8_t *data, int len);`

- **参数说明**：无

- **返回值**：无

### flush

- **接口功能**：刷新串口缓冲区，清空未处理数据

- **接口声明**：`void flush() override;`

- **参数说明**：无

- **返回值**：无

### enumerate\_devices

- **接口功能**：枚举当前系统中所有可用的串口设备名称

- **接口声明**：`static std::vector<std::string> enumerate_devices();`

- **参数说明**：无

- **返回值**：无

### 成员变量说明

仅文档描述

---

## EtherCatCommunication 接口文档

### 接口功能

提供基于 EtherCAT 的实时通信能力，包括打开 / 关闭 EtherCAT 网络、PDO/SDO 的读写、错误处理、线程管理以及可用网卡枚举功能。

### 构造函数

- **接口功能**：创建一个 EtherCAT 通信对象，并指定使用的网络接口名称

- **接口声明**：`EtherCatCommunication(const std::string if_name);`

- **参数说明**：无

- **返回值**：无

### open

- **接口功能**：打开 EtherCAT 网络，初始化主站、扫描从站、创建 PDO 线程和错误监控线程

- **接口声明**：`ErrorCode open() override;`

- **参数说明**：无

- **返回值**：

`SUCCESS`：成功 

`DEVICE_NOT_INIT`：初始化失败或未检测到从站 

`OPEN_DEVICE_FAILED`：重复打开

### close

- **接口功能**：关闭 EtherCAT 主站，停止所有线程并释放资源

- **接口声明**：`ErrorCode close() override;`

- **参数说明**：无

- **返回值**：

`SUCCESS`：成功 

`ERROR_COMMUNICATION`：设备未打开

### send（SDO 写）

- **接口功能**：使用 SDO 方式向指定从站写入数据（基于固定对象字典）

- **接口声明**：`ErrorCode send(uint8_t device_id, const uint8_t *data, int len) override;`

- **参数说明**：

`device_id`：从站 ID（1 \~ slavecount） 

`data`：待发送数据 

`len`：数据长度

- **返回值**：

`SUCCESS`：成功 

`INVALID_DEVICE_ID`：设备或参数错误

### receive（SDO 读）

- **接口功能**：使用 SDO 方式从指定从站读取数据（基于固定对象字典）

- **接口声明**：`size_t receive(uint8_t device_id, uint8_t *data, int len) override;`

- **参数说明**：

`device_id`：从站 ID 

`data`：接收缓冲区 

`len`：期望读取长度（最大 256 字节）

- **返回值**：

`>0`：实际读取字节数 

`0`：失败

### pdo\_read

- **接口功能**：从指定从站读取 PDO 输入数据（来自 IOMap）

- **接口声明**：`size_t pdo_read(uint8_t device_id, uint8_t *data, size_t len) override;`

- **参数说明**：

`device_id`：从站 ID 

`data`：接收缓冲区 

`len`：读取长度（不得超过从站输入字节数）

- **返回值**：

`>0`：实际读取字节数 

`0`：失败

### pdo\_write

- **接口功能**：向指定从站写入 PDO 输出数据（写入 IOMap，由内部线程周期发送）

- **接口声明**：`ErrorCode pdo_write(uint8_t device_id, uint8_t *data, size_t len) override;`

- **参数说明**：

`device_id`：从站 ID 

`data`：发送数据 

`len`：数据长度（不得超过从站输出字节数）

- **返回值**：

`SUCCESS`：成功 

`ERROR_COMMUNICATION_BUSY`：参数错误或设备无效

### flush

- **接口功能**：EtherCAT 不需要手动 flush，因此此接口为空实现

- **接口声明**：`void flush() override {}`

- **参数说明**：无

- **返回值**：无

### slave\_number

- **接口功能**：获取当前扫描到的从站数量

- **接口声明**：`int slave_number();`

- **参数说明**：无

- 返回值：

从站数量

### set\_sdo\_callback

- **接口功能**：添加一个 SDO/PDO 回调函数，用于处理输入 / 输出数据

- **接口声明**：`void set_sdo_callback(const PDO_CallBack &func);`

- **参数说明**：

`func`：回调函数（在 PDO 周期线程中调用）

- **返回值**：无

### enumerate\_devices（静态函数）

- **接口功能**：枚举系统所有可用于 EtherCAT 的网卡接口

- **接口声明**：`static std::vector<std::string> enumerate_devices();`

- **返回值**： 网卡名称列表（仅包含检测到目标从站的接口）

### 内部成员变量说明

仅文档描述

---

## GreenHand API 接口说明

### 接口功能

GreenHand 是基于 BaseHand 的高级封装，针对 Green 手设备提供用户级接口，包括：

- 关节控制（用户单位 → 编码值自动转换） 

- 关节状态读取（自动解码） 

- 传感器数据读取（含特殊掌心传感器） 

- EtherCAT PDO 实时通信支持 

- 串口/CAN 帧通信兼容 

支持两种工作模式：

EtherCAT 模式（PDO 实时驱动）

串口 / CAN 模式（请求\-响应\)

## 构造函数（GreenHand）

- 接口功能：创建 GreenHand 对象，并自动注册 EtherCAT PDO 回调（若使用 EtherCAT） 

- 接口声明： 

GreenHand\(uint8\_t device\_id, std::shared\_ptr\<CommunicationBase\> comm\);

- 注意事项： 

继承自 BaseHand，会自动执行设备映射初始化 

当通信类型为 EtherCAT 时： 

- 自动注册 PDO 回调函数 

- 数据通过 `flush_pdo()` 实时更新 

若 comm 为空，会抛异常（由 BaseHand 处理） 

- 返回值：无

## flush\_downlink\_data

- 接口功能：发送当前控制命令并刷新设备数据（串口/CAN 模式下生效） 

- 接口声明： 

ErrorCode flush\_downlink\_data\(\);

- 注意事项： 

EtherCAT 模式下： 

- 此函数为空实现（直接返回 SUCCESS） 

- 数据通过 PDO 自动更新 

串口/CAN 模式下： 

- 发送完整控制帧 

- 接收并更新 uplink\_data\_ 

内部使用写锁保护缓冲区 

返回值：ErrorCode

## set\_joint\_cmd（单关节）

- 接口功能：设置单个关节控制命令（用户单位） 

- 接口声明： 

ErrorCode set\_joint\_cmd\(int joint\_id, const JointCommand\_User\& joint\_data\);

- 注意事项： 

joint\_id 范围：0 ≤ joint\_id \< JOINT\_NUM 

自动完成： 

- 单位转换（kp/kd/position/speed/torque） 

- 写入 downlink\_data\_ 

最终调用 flush\_downlink\_data\(\) 发送 

线程安全（写锁） 

- 返回值：ErrorCode

## set\_joint\_cmd（批量）

- 接口功能：批量设置所有关节控制命令 

- 接口声明： 

ErrorCode set\_joint\_cmd\(const std::vector\<JointCommand\_User\>\& joint\_data\_list\);

- 注意事项： 

按顺序写入 joint\_cmds 

超出 JOINT\_NUM 的部分会被忽略 

最后统一发送（flush） 

- 返回值：ErrorCode 

## get\_joint\_state\_user（单关节）

- 接口功能：获取指定关节状态（用户单位） 

- 接口声明： 

`JointState_User get_joint_state_user(int joint_id, ErrorCode &error_code, bool force_flush = false);`

- 注意事项： 

可选 force\_flush： 

- true → 先触发一次通信刷新 

自动完成解码： 

- position / speed / torque 转换 

自动错误码转换（硬件码 → 自定义码） 

线程安全（读锁） 

返回值：JointState\_User

## get\_joint\_state\_user（全部关节）

- 接口功能：获取所有关节状态 

- 接口声明： 

`std::vector<JointState_User> get_joint_state_user(ErrorCode &error_code, bool force_flush = false);`

- 注意事项： 

遍历所有关节 

每个关节独立解码 

EtherCAT 模式下为实时数据 

- 返回值：关节状态数组

## get\_sensor\_state（单传感器）

- 接口功能：获取指定传感器原始数据 

- 接口声明： `Sensor get_sensor_state(int sensor_id, bool force_flush = false);`

- 注意事项： 

sensor\_id 必须存在于映射表 

返回原始结构（未解码） 

线程安全（读锁） 

- 返回值：Sensor

## get\_sensor\_state（全部）

- 接口功能：获取全部传感器数据 

- 接口声明： 

`std::vector<Sensor> get_sensor_state(bool force_flush = false);`

- 注意事项： 

返回内部数组拷贝 

不做数据转换 

- 返回值：Sensor 数组

## get\_sensor\_state\_user（单传感器）

- 接口功能：获取用户级传感器数据（含错误码转换） 

- 接口声明： 

`Sensor_User get_sensor_state_user(int sensor_id, bool force_flush = false);`

- 注意事项： 

自动转换错误码 

数据结构转换为用户格式 

- 返回值：Sensor\_User 

## get\_sensor\_state\_user（全部）

- 接口功能：获取全部用户级传感器数据 

- 接口声明： 

`std::vector<Sensor_User> get_sensor_state_user(bool force_flush = false);`

- 注意事项： 

遍历所有传感器 

自动错误码映射 

- 返回值：Sensor\_User 数组

## get\_sensor\_state2（特殊传感器）

- 接口功能：获取掌心（ID=30）传感器数据 

- 接口声明： 

`Sensor_2 get_sensor_state2(int sensor_id, bool force_flush = false);`

- 注意事项： 

该传感器占用两个 PDO 槽位 

数据已在 PDO 回调中拼接完成 

- 返回值：Sensor\_2

## get\_sensor\_state2\_user

- 接口功能：获取用户级掌心传感器数据 

- 接口声明： 

`Sensor_2_User get_sensor_state2_user(int sensor_id, bool force_flush = false);`

- 注意事项： 

自动错误码转换 

返回完整13通道数据 

- 返回值：Sensor\_2\_User

## flush\_pdo（内部回调）

- 接口功能：EtherCAT PDO 数据收发与解析（内部使用） 

- 接口声明： 

void flush\_pdo\(int slave\_id, uint8\_t\* output, size\_t output\_size,

uint8\_t\* input, size\_t input\_size\);

- 注意事项（重要）： 

**仅 EtherCAT 模式生效**

1. 功能包含： 

    - 下行：发送3个关节命令（轮询） 

    - 上行：接收3个关节状态 \+ 3个传感器 

2. 特殊处理： 

    - Sensor ID 30（掌心）拆包重组 

3. 使用读写锁保护数据 

- 返回值：无

## 编解码接口（内部）

### encode\_joint\_command

- 功能：用户指令 → 设备编码 

### decode\_joint\_state

- 功能：设备数据 → 用户数据 

### decode\_sensor / decode\_sensor\_2

- 功能：传感器数据解码 \+ 错误码映射

## 注意事项（整体）

**通信模式差异**

- EtherCAT：实时 PDO，无需手动 flush 

- 串口/CAN：必须调用 flush\_downlink\_data 

**线程安全**

- 写操作：unique\_lock 

- 读操作：shared\_lock 

**错误码机制**

- 硬件错误码 → ErrorManager → 自定义错误码 

**性能特点**

- PDO：1ms 实时周期 

- 串口：请求\-响应模式

---

# 数据结构

- JointCommand

- JointCommand\_User

- JointState\_User

- JointState\_User

- Sensor（类型 1）

- Sensor\_2（类型 2）

- 常量



---

# 关节 ID 说明

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODNkZmRjZmY2NmFiZTE2MTI2ZWI5NGQ4MmZhMTEyOWNfM2ZjNjEyMjE2ZTI3MDRhYWE5NmViYmMyYzI5YmYyMjdfSUQ6NzYyMDEyMTQxMjQ3NzIwOTc4MF8xNzgwNTU3MzQ0OjE3ODA2NDM3NDRfVjM)

上图所示，XHAND 1 Pro灵巧手由一个掌板、21个指关节板，指关节板附带18个触觉传感器组成。每一个板由唯一ID作为通信的唯一标识，且不重复。

上图中：

- 红色数字，指示的是关节板的ID号

- 黑色数字，指示的是触觉传感器的ID号

- 关节板的ID  和 触觉传感器的ID号 相同，表示两者所处的位置相同

- 关节板的ID为 0  4  8  12 16  20 的指关节没有对应的触觉传感器

- 触觉传感器的ID号为 30  31  32 的是手掌心的触觉传感器

---

# 灵巧手 ID

- 灵巧手 ID **默认为 0**，用户可修改，范围：**0\~125**

- 掌板 ID 是灵巧手 ID 的最高位置 1，用于和指关节 ID、指尖传感器 ID 区分

- 示例：灵巧手 ID 设置为 3（0x03），实际通讯使用 **0x83** 作为掌板 ID

