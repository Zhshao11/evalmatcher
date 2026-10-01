# =============================================================================
# EvalMatcher 服务端镜像
# -----------------------------------------------------------------------------
# 设计要点：
#   1) 权重与数据集**不进镜像**：它们由 docker-compose 从宿主机只读挂载到
#      /app/weights 与 /app/data。换权重不用重新 build。
#   2) 基础镜像用 nvidia/cuda runtime（CUDA 12.8 + cuDNN），torch 从 PyTorch
#      官方 CUDA 源装，版本用 ARG 控制，方便对齐宿主机驱动。
#      若你本地已有合适的 pytorch 镜像，可以：
#         docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
#     这样会跳过 pip 装 torch（脚本会自动检测）。
#   3) 第三方算法代码（third/）整包进镜像，它是算法本体，不是权重。
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG CUDA_TAG=cu128
ARG DEBIAN_FRONTEND=noninteractive

# ---- 系统库：opencv 需要 libgl/libglib，pydegensac 可能要现场编译 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        build-essential cmake git \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && python3 -m pip install --no-cache-dir -U pip setuptools wheel

WORKDIR /app

# ---- torch：只有基础镜像里没有 torch 时才装（换成 pytorch 官方镜像可省这一步）----
# 注意 --index-url 指向 PyTorch 官方 CUDA 源，装出来才是带 CUDA 的 wheel。
RUN python -c "import torch" 2>/dev/null \
    || python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/${CUDA_TAG} \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}

# ---- 其余依赖 ----
COPY server/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# ---- 代码 ----
COPY server/ /app/
COPY config/ /app/config/

# 容器内的固定挂载点（真实内容由 compose 挂进来；这里只建空壳，保证目录存在）
RUN mkdir -p /app/weights /app/data /app/output

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVER_CONFIG=/app/config/server.yaml

EXPOSE 8000

# entrypoint：先建权重软链 + 六项启动检查，全过才起服务。
# 任何一项不过就立刻退出并打印"缺什么、怎么做"，不会带病启动。
COPY server/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

HEALTHCHECK --interval=30s --timeout=8s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5).status==200 else 1)"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "app.py"]
