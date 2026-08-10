#!/bin/bash
# 启动 cloud TensorBoard。
# 为什么是独立脚本: 经 SSH 用 inline `bash -c "source...conda...tensorboard"` 会 255(通道挂);
# 脚本形式 + `( setsid bash scripts/start_tb.sh >log 2>&1 </dev/null & )` 跟 master 一样能干净 detach。
cd /root/era_dexhand_lab || exit 1
source scripts/cloud_lib.sh
cloud_activate_conda
pkill -f "[t]ensorboard" 2>/dev/null
sleep 1
exec tensorboard --logdir logs/debug --port 6006 --bind_all
