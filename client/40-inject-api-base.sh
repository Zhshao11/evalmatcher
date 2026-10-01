#!/bin/sh
# =============================================================================
# 前后端分开部署时：把外部 API 地址注入前端。
#
# 同源部署（默认）不用管：前端请求 /api，由 nginx 反代。
# 只有把前端单独放到别的域名下时才需要设 EXTERNAL_API_BASE，例如：
#     EXTERNAL_API_BASE=https://api.example.com
# 这里把它写成 window.__EM_API_BASE__，index.html 里的 API_BASE 会优先读它。
# =============================================================================
set -e

HTML=/usr/share/nginx/html/index.html

if [ -n "${EXTERNAL_API_BASE:-}" ]; then
  if ! grep -q '__EM_API_BASE__' "$HTML"; then
    # 把 / 和 & 转义，避免 sed 替换时被当成特殊字符
    ESC=$(printf '%s' "$EXTERNAL_API_BASE" | sed -e 's/\\/\\\\/g' -e 's/&/\\&/g' -e 's|/|\\/|g')
    sed -i "s#</head>#<script>window.__EM_API_BASE__=\"${ESC}\";</script></head>#" "$HTML"
    echo "[client] 已注入外部 API 地址：${EXTERNAL_API_BASE}"
  fi
fi
