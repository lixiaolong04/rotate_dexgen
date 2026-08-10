#!/usr/bin/env bash
# 进入 Xhand 部署容器。把整个项目仓库挂载到 /workspace(改代码即时生效, 不用重建镜像)。
# 串口设备默认 /dev/ttyUSB0, 可传第一个参数覆盖; 设备不存在(真机未接)时自动跳过透传, 仍可进容器跑 --dry_run。
#
# 用法:
#   ./run.sh                 # 优先真机镜像；没有时使用离线镜像
#   ./run.sh /dev/ttyACM0    # 指定串口设备
#   XHAND_DEPLOY_IMAGE=xhand-deploy:offline ./run.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DEV="${1:-/dev/ttyUSB0}"
IMAGE="${XHAND_DEPLOY_IMAGE:-}"

if [ -z "$IMAGE" ]; then
    if docker image inspect xhand-deploy:22.04 >/dev/null 2>&1; then
        IMAGE="xhand-deploy:22.04"
    elif docker image inspect xhand-deploy:offline >/dev/null 2>&1; then
        IMAGE="xhand-deploy:offline"
    else
        echo "❌ 未找到部署镜像。无真机时先运行: $HERE/build.sh --offline"
        exit 1
    fi
fi

if [ "$IMAGE" = "xhand-deploy:offline" ] && [ -e "$DEV" ]; then
    echo "❌ 离线镜像不包含厂商 SDK，拒绝透传真机设备 $DEV"
    echo "   请准备 xhandSDK 后构建 xhand-deploy:22.04"
    exit 1
fi

DEV_ARGS=()
if [ -e "$DEV" ]; then
    DEV_ARGS=(--device="$DEV")
    echo "[run] 透传串口设备: $DEV"
else
    echo "[run] ⚠️ $DEV 不存在(真机未接) -> 不透传设备; 容器内可跑 --dry_run 离线验证"
fi
echo "[run] 镜像: $IMAGE"

docker run -it --rm \
    "${DEV_ARGS[@]}" \
    -v "$REPO":/workspace \
    -w /workspace \
    "$IMAGE" bash
