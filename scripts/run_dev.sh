#!/usr/bin/env bash
# =============================================================================
# 非 Docker 的开发启动方式（沿用改造前的用法：一个 Python 进程跑起来）
#
#   ./scripts/run_dev.sh                 # 用默认端口 8000
#   PORT=8300 ./scripts/run_dev.sh       # 换端口
#   GPU=3 ./scripts/run_dev.sh           # 换 GPU
#
# 与 Docker 部署的区别只有一点：路径不再来自容器挂载，而来自下面三个环境变量
# （不设就走 config/server.yaml 里写的容器路径 /app/*，那在非容器环境里不存在）。
#
#   EM_WEIGHTS_DIR=~/evalmatcher-weights \
#   EM_DATASET_DIR=~/evalmatcher-data/VIS_SAR \
#   EM_OUTPUT_DIR=~/evalmatcher-output \
#   ./scripts/run_dev.sh
#
# 前端：直接打开 client/index.html 是不行的（它要 fetch /api）。
# 开发时另开一个终端跑：
#   python -c "import http.server" ... 或者干脆 docker compose up client
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PORT="${PORT:-}"
GPU="${GPU:-0}"
PY="${PY:-python3}"

cd "$REPO/server"

export SERVER_CONFIG="${SERVER_CONFIG:-$REPO/config/server.yaml}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[dev] repo=$REPO config=$SERVER_CONFIG gpu=$GPU"
[ -n "$PORT" ] && export PORT="$PORT"

# 启动检查：配置 / 权重 / 数据集 / 输出目录 / CUDA / 算法，缺哪项都会在这里说清楚
"$PY" preflight.py

exec "$PY" app.py
