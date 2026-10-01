#!/usr/bin/env bash
# =============================================================================
# 下载权重与数据集
#
# 仓库里**没有**权重和数据集（体积太大），它们放在 Google Drive 上：
#   数据集  https://drive.google.com/file/d/1DHfI1j4yELughX-ljwvNofgofMS8lHU2/view
#   权重    https://drive.google.com/file/d/1Y6D-Fu8-99f7buNCag1tH4C_JVO059Cy/view
#
# 用法（默认就是上面这两个，直接跑即可）：
#   pip install gdown          # Google Drive 大文件需要它（见注释）
#   ./scripts/fetch_assets.sh
#
# 只想下其中一个：
#   ONLY=weights ./scripts/fetch_assets.sh
#   ONLY=dataset ./scripts/fetch_assets.sh
#
# 换自己的链接（Google Drive 文件 ID，或任意直链）：
#   WEIGHTS_GDRIVE_ID=<id> DATASET_GDRIVE_ID=<id> ./scripts/fetch_assets.sh
#   WEIGHTS_URL=https://... DATASET_URL=https://... ./scripts/fetch_assets.sh
#
# 为什么需要 gdown：Google Drive 对大文件有"病毒扫描确认页"，
# curl/wget 直接下会拿到一个 HTML 页面而不是真正的文件，gdown 会处理这一步。
# 不想装就手动用浏览器下载，再解压到对应目录。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
DATASET_DIR="${DATASET_DIR:-$REPO/data}"
ONLY="${ONLY:-all}"

# 默认：仓库维护者提供的 Google Drive 文件 ID
DATASET_GDRIVE_ID="${DATASET_GDRIVE_ID:-1DHfI1j4yELughX-ljwvNofgofMS8lHU2}"
WEIGHTS_GDRIVE_ID="${WEIGHTS_GDRIVE_ID:-1Y6D-Fu8-99f7buNCag1tH4C_JVO059Cy}"
DATASET_URL="${DATASET_URL:-}"
WEIGHTS_URL="${WEIGHTS_URL:-}"

WURL="https://drive.google.com/file/d/${WEIGHTS_GDRIVE_ID}/view"
DURL="https://drive.google.com/file/d/${DATASET_GDRIVE_ID}/view"

have() { command -v "$1" >/dev/null 2>&1; }

need_gdown() {
  have gdown && return 0
  echo "[缺失] 下载 Google Drive 大文件需要 gdown：pip install gdown"
  echo "       也可以手动打开链接用浏览器下载：$1"
  return 1
}

# 解压：**按内容判断类型，不看文件名后缀** ——
# 下载时统一落成 .download_part（没有扩展名），看后缀永远会走到最后的兜底分支。
unpack() {
  local arc="$1" dest="$2"
  mkdir -p "$dest"
  if tar -tf "$arc" >/dev/null 2>&1; then
    tar -xf "$arc" -C "$dest"          # tar / tar.gz / tar.xz 等 tar 都能自动识别
  elif have unzip && unzip -l "$arc" >/dev/null 2>&1; then
    unzip -q -o "$arc" -d "$dest"
  else
    mv "$arc" "$dest/downloaded_asset"  # 不是压缩包：本身就是要的文件
  fi

  # 包里只有一个顶层目录（如 weights/ours/...）时，把内容上移一层
  local subs files only_dir
  subs=$(find "$dest" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | wc -l | tr -d ' ')
  files=$(find "$dest" -mindepth 1 -maxdepth 1 -type f ! -name '.*' | wc -l | tr -d ' ')
  if [ "$subs" = "1" ] && [ "$files" = "0" ]; then
    only_dir=$(find "$dest" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | head -1)
    echo "[整理] 抹平顶层目录：$(basename "$only_dir")/"
    tmp2="$(mktemp -d)"
    (cd "$only_dir" && tar -cf - .) | (cd "$tmp2" && tar -xf -)
    rm -rf "$only_dir"
    (cd "$tmp2" && tar -cf - .) | (cd "$dest" && tar -xf -)
    rm -rf "$tmp2"
  fi
}

fetch() {
  local dest="$1" url="$2" gurl="$3"
  mkdir -p "$dest"
  local tmp="$dest/.download_part"
  rm -f "$tmp"
  if [ -n "$url" ]; then                       # 任意直链
    echo "[下载] $url"
    if have curl; then curl -fL --retry 3 -o "$tmp" "$url"
    elif have wget; then wget -O "$tmp" "$url"
    else echo "需要 curl 或 wget" >&2; return 1; fi
  else                                          # Google Drive
    need_gdown "$gurl" || return 1
    echo "[下载] $gurl"
    gdown --fuzzy "$gurl" -O "$tmp" || gdown "$gurl" -O "$tmp"
  fi
  [ -s "$tmp" ] || { echo "[失败] 下载为空：$gurl"; return 1; }
  unpack "$tmp" "$dest"
  rm -f "$tmp"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "weights" ]; then
  fetch "$WEIGHTS_DIR" "$WEIGHTS_URL" "$WURL"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "dataset" ]; then
  fetch "$DATASET_DIR" "$DATASET_URL" "$DURL"
fi

# 关键：统一写成 ${VAR}。$VAR 后面紧跟全角字符（如这里的冒号）会被 bash
# 当成变量名的一部分，set -u 下直接报 unbound variable —— 踩过一次。
echo
echo "===== 目录检查 ====="
echo "权重 ${WEIGHTS_DIR} ："
find "$WEIGHTS_DIR" -maxdepth 2 -type f 2>/dev/null | sed 's|^|  |' | head -15
echo "数据集 ${DATASET_DIR} ："
ls -d "$DATASET_DIR"/VIS_SAR/test/{VIS,SAR,transforms} 2>/dev/null | sed 's|^|  |' \
  || echo "  （没找到 VIS_SAR/test/{VIS,SAR,transforms}，请检查压缩包结构）"
echo
echo "把这两个绝对路径填进 .env 的 WEIGHTS_DIR / DATASET_DIR："
echo "  WEIGHTS_DIR=$(cd "$WEIGHTS_DIR" 2>/dev/null && pwd)"
echo "  DATASET_DIR=$(cd "$DATASET_DIR" 2>/dev/null && pwd)"
