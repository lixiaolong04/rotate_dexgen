# Overview
This is a repo for reinforcement learning sim2real rotation demo on SharpaWave, provides a step-by-step guide for training, visualizing and deploying.

<p align="center">
  <img src="resources/sim.gif" width="45%" />
  <img src="resources/real.gif" width="45%" />
</p>

# 1. Environment Setup
## 1.1. Follow the official  Isaaclab installation guide:
Install [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html). 

Ubuntu 22.04, conda environment, release/2.2.0 and release/2.3.0 have been tested.

CAUTION⚠️: A minimum of 32GB RAM is required. For specific requirements, please refer to [requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
## 1.2. Install this repo:  
```bash
conda activate env_isaaclab 
cd sharpa_tac_rl 
pip install -e .
```

# 2. Training
```bash
export CUDA_VISIBLE_DEVICES=0
```
## 2.1. Generate grasp cache
```bash
# CAUTION⚠️: Same object scale config will overwrite the older one.
python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Xhand-Wave-v0 --headless
```
## 2.2. Train the policy
```bash
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Xhand-Wave-v0 --headless --num_envs 10
```
## 2.3. Distillation
```bash
# last.pth is recommended if curriculum is enabled
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --algorithm ProprioAdapt --load_path ${pth}
```

# 3. Visualization
## 3.1. Visualize trained policy
```bash
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --load_path ${pth}
```
## 3.2. Visualize distillated policy
```bash
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --algorithm ProprioAdapt --load_path ${pth}
```

# 4. Deploy
## 4.1. Prepare SharpaWave and object
1. Calibrate SharpaWave through SharpaPilot. 
2. A cylinder with radius of 24mm and height of 60mm via 3D priting is recommended under default configuration.
## 4.2. Deploy on SharpaWave (HostComputer Tactile, Recommended)
### 4.2.1. Configure docker
```bash
# INFOℹ️: Install docker and nvidia-ctk following steps 1-4 in <Steps to Acquire 180 Hz High-Frame-Rate High-Performance Tactile Information>.
cd rl_isaaclab/utils
# Configure docker-compose, substitute ${sharpa-rl-lab} with this repo path.
xhost +local:root
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose up -d
docker exec -it sharpawave_rl_dev bash
rm -r ~/sharpawave-rl-lab/rl_isaaclab/utils/python/
cp -r ~/sharpa-wave-sdk/python/sharpa/ ~/sharpawave-rl-lab/rl_isaaclab/utils/python/
cd ~/sharpawave-rl-lab/
python3 -m pip install -e .
```
### 4.2.2. Deploy
```bash
# INFOℹ️: Keyboard control is enabled by default. Press 'e' to start, press 'w' to freeze, press 'q' to go home.
python3 rl_isaaclab/scripts/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --hand_side ${0/1} --load_path ${pth}
```
## 4.3. Deploy on SharpaWave (OnBoard Tactile)
### 4.3.1. Configure SharpaWaveSDK (For Deploy)
```bash
# INFOℹ️: Install SharpaWaveSDK following the official user manual. ${SharpaWaveSDK} is the root path of the SDK.
rm -r rl_isaaclab/utils/python
cp -r ${SharpaWaveSDK}/python rl_isaaclab/utils/python
```
### 4.3.2. Deploy
```bash
# INFOℹ️: Keyboard control is enabled by default. Press 'e' to start, press 'w' to freeze, press 'q' to go home.
python rl_isaaclab/scripts/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --enable_on_board --hand_side ${0/1} --load_path ${pth}
```

# 5. Configure your own task via modifying the config file
Please refer to rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env_cfg.py and rl_isaaclab/tasks/inhand_rotate/sharpa_wave_deploy_env_cfg.py for details.


Joint 名称	对应手指	对应位置	含义
right_index_MCP_FE	食指	掌指关节	指根弯曲/伸直
right_index_MCP_AA	食指	掌指关节	指根外展/内收
right_index_PIP	食指	近端指间关节	中间关节
right_index_DIP	食指	远端指间关节	指尖前关节
right_middle_MCP_FE	中指	掌指关节	指根弯曲/伸直
right_middle_MCP_AA	中指	掌指关节	指根外展/内收
right_middle_PIP	中指	近端指间关节	中间关节
right_middle_DIP	中指	远端指间关节	指尖前关节
right_ring_MCP_FE	无名指	掌指关节	指根弯曲/伸直
right_ring_MCP_AA	无名指	掌指关节	指根外展/内收
right_ring_PIP	无名指	近端指间关节	中间关节
right_ring_DIP	无名指	远端指间关节	指尖前关节
right_pinky_CMC	小指	腕掌关节	小指根部更靠掌部
right_pinky_MCP_FE	小指	掌指关节	指根弯曲/伸直
right_pinky_MCP_AA	小指	掌指关节	指根外展/内收
right_pinky_PIP	小指	近端指间关节	中间关节
right_pinky_DIP	小指	远端指间关节	指尖前关节
right_thumb_CMC_FE	拇指	腕掌关节	拇指根部弯曲/伸直
right_thumb_CMC_AA	拇指	腕掌关节	拇指根部外展/内收
right_thumb_MCP_FE	拇指	掌指关节	拇指中段弯曲/伸直
right_thumb_MCP_AA	拇指	掌指关节	拇指中段外展/内收
right_thumb_IP	拇指	指间关节	拇指末端关节