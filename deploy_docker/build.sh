#!/usr/bin/env bash
# 构建 Xhand 部署镜像。
#
#   ./build.sh            构建含旧版厂商 SDK 的真机镜像
#   ./build.sh --offline  构建 FakeHW 离线验证镜像，不需要厂商 SDK
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SDK="$HERE/../xhandSDK"
STAGE="$HERE/.build_ctx"

if [ "${1:-}" = "--offline" ]; then
    echo "[build] 构建无厂商 SDK 的离线验证镜像 xhand-deploy:offline"
    docker build -f "$HERE/Dockerfile.offline" -t xhand-deploy:offline "$HERE"
    echo "✅ 已构建镜像 xhand-deploy:offline（仅允许 --dry_run）"
    exit 0
fi
if [ "$#" -ne 0 ]; then
    echo "用法: $0 [--offline]"
    exit 2
fi

# 自动定位 SDK 文件(容忍文件名里的空格 / (1) / 版本号变化)
if [ ! -d "$SDK" ]; then
    echo "❌ 缺少 $SDK"
    echo "   真机镜像需要旧版 hand_sdk_*.deb 和解包后的 hand_controller 包。"
    echo "   无真机时可先运行: $0 --offline"
    exit 1
fi
DEB="$(find "$SDK" -maxdepth 1 -name 'hand_sdk_*.deb' | head -1)"
PKG="$(find "$SDK" -maxdepth 2 -type d -name 'hand_controller' | head -1)"

if [ -z "$DEB" ]; then echo "❌ 没找到 xhandSDK/hand_sdk_*.deb"; exit 1; fi
if [ -z "$PKG" ]; then echo "❌ 没找到 xhandSDK 下解包的 hand_controller 包目录"; exit 1; fi
echo "[build] deb = $DEB"
echo "[build] pkg = $PKG"

rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$HERE/Dockerfile" "$STAGE/Dockerfile"
cp "$DEB" "$STAGE/hand_sdk.deb"
cp -r "$PKG" "$STAGE/hand_controller"

docker build -t xhand-deploy:22.04 "$STAGE"
rm -rf "$STAGE"
echo "✅ 已构建镜像 xhand-deploy:22.04"
