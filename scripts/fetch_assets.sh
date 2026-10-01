#!/usr/bin/env bash
# =============================================================================
# 下载权重与数据集（网盘直链版）
#
# 仓库里**没有**权重和数据集（体积太大，且不便于二次分发）。
# 用法（填入网盘直链后）：
#   WEIGHTS_URL="https://pan.example.com/d/xxxx" \
#   DATASET_URL="https://pan.example.com/d/yyyy" \
#   ./scripts/fetch_assets.sh
#
# 也可以手动下载后解压，只要目录结构对得上（见下），脚本不是必须的。
#
# 期望的目录结构
# --------------
# $WEIGHTS_DIR/
#   ours/     2024_10_10-10_44_34_VIS_SAR_106000.pth
#             2024_10_21-10_14_50_VIS_IR_62500.pth
#             2024_11_02-09_43_00_VIS_NIR_30000.pth
#   redfeat/  VIS_SAR.pth  VIS_IR.pth  VIS_NIR.pth
#   d2net/    d2_tf.pth
#   minima/   weights_xoftr_640.ckpt  minima_loftr.ckpt
#
# $DATASET_DIR/
#   VIS_SAR/test/VIS/          424 张 .png
#   VIS_SAR/test/SAR/          424 张同名 .png
#   VIS_SAR/test/transforms/   424 个 .mat（1.png 对应 1.png.12.mat / .21.mat）
#
# 文件名与 config/server.yaml 的 methods[*].weights[].file 一一对应。
# 想改文件名就改 server.yaml，两边对齐即可。
# =============================================================================
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-./weights}"
DATASET_DIR="${DATASET_DIR:-./data}"
WEIGHTS_URL="${WEIGHTS_URL:-}"
DATASET_URL="${DATASET_URL:-}"

have() { command -v "$1" >/dev/null 2>&1; }

download() {
  local url="$1" out="$2"
  if [ -z "$url" ]; then
    echo "[skip] 未提供链接：$out"
    return 0
  fi
  echo "[down] $out <- $url"
  if have curl; then curl -fL --retry 3 -o "$out" "$url"
  elif have wget; then wget -O "$out" "$url"
  else echo "需要 curl 或 wget" >&2; return 1; fi
}

mkdir -p "$WEIGHTS_DIR" "$DATASET_DIR"

[ -n "$WEIGHTS_URL" ] && download "$WEIGHTS_URL" "${WEIGHTS_DIR}.tar" || true
[ -n "$DATASET_URL" ] && download "$DATASET_URL" "${DATASET_DIR}.tar" || true

# 下载到的是 tar 包就解开（.zip 用 unzip）
for t in "${WEIGHTS_DIR}.tar" "${DATASET_DIR}.tar"; do
  [ -f "$t" ] || continue
  if [[ "$t" == "${WEIGHTS_DIR}.tar" ]]; then dest="$WEIGHTS_DIR"; else dest="$DATASET_DIR"; fi
  echo "[untar] $t -> $dest"
  mkdir -p "$dest"
  tar -xf "$t" -C "$dest" --strip-components=1
  rm -f "$t"
done

echo
echo "目录检查："
echo "  权重：$WEIGHTS_DIR"
ls -la "$WEIGHTS_DIR" 2>/dev/null || true
echo "  数据集：$DATASET_DIR"
ls -la "$DATASET_DIR" 2>/dev/null || true
echo
echo "然后把这两个绝对路径填进 .env 的 WEIGHTS_DIR / DATASET_DIR。"
