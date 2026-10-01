#!/usr/bin/env bash
# =============================================================================
# 服务端入口：启动检查 -> 权重软链 -> 起服务
#
# 为什么不直接 CMD python app.py：
#   权重/数据集/输出目录都是挂载进来的，挂载没配对时服务能起、但一点"开始评测"
#   就 FileNotFoundError。所以启动前把六项全部检查一遍，不过就退出（exit 1），
#   让 docker compose ps / logs 一眼看到原因。
# =============================================================================
set -euo pipefail

cd /app

if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
  echo "[entrypoint] SKIP_PREFLIGHT=1，跳过启动检查（出问题别怪我没提醒）"
else
  python preflight.py
fi

exec "$@"
