# EvalMatcher

跨模态图像匹配（VIS–SAR）在线评测平台：浏览器点一下，后端**真跑** 5 种匹配方法，
逐对回传进度与指标，边跑边出图。

- **client**：零依赖单页站点（HTML + 手写 Canvas 图表），由 nginx 提供，并把 `/api` 反代到服务端
- **server**：Flask + PyTorch，跑 d2net / ReDFeat / **HLDD** / XoFTR / LoFTR，输出 SR / NCM / ME / RMSE / MMA / Homography 等指标

```
浏览器 ──► client(nginx :80) ──/api──► server(Flask :8000) ──► GPU
                                              │
              /app/weights(ro)  /app/data(ro)  /app/output(rw)
                   ▲                 ▲              ▲
              .env 里填的宿主机目录（权重 / 数据集 / 结果）
```

> 权重与数据集**不在仓库里**，也不进 Docker 镜像，需要单独准备（见 [权重与数据集](#4-权重与数据集))。

---

## 1. 环境要求

### 硬件

| 场景 | 要求 | 说明 |
|---|---|---|
| GPU（推荐，正式评测） | NVIDIA GPU，显存 ≥ 8 GB，驱动支持 CUDA 12.x | 实测环境：RTX 5090 32 GB，5 方法 × 424 对约 6 分钟 |
| CPU（仅冒烟） | 任意多核 CPU + ≥ 16 GB 内存 | 只能跑几对验证链路，全量 424 对会非常慢 |

### 软件

| 组件 | 版本 | 用途 |
|---|---|---|
| Docker | ≥ 24（含 Compose v2 插件） | 构建与编排 |
| NVIDIA 驱动 | ≥ 525（CUDA 12.x） | GPU 模式必需 |
| NVIDIA Container Toolkit | 最新版 | 让容器能用 GPU |
| Git | 任意 | 克隆仓库 |

NVIDIA Container Toolkit 安装（Ubuntu 为例）：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
# 验证：应该能看到本机 GPU
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

> 只想跑 CPU 冒烟：把 `.env` 的 `DEVICE` 改成 `cpu`，并删掉 `docker-compose.yml` 里
> server 的 `deploy.resources.reservations.devices` 整段，再 `docker compose up -d --build`。

---

## 2. 从全新克隆到启动成功

```bash
# ① 克隆
git clone git@github.com:Zhshao11/evalmatcher.git
cd evalmatcher

# ② 下载权重与数据集（不在仓库里，约 324 MB，见第 4 节）
pip install gdown                  # Google Drive 大文件需要它
./scripts/fetch_assets.sh          # 落到仓库下的 weights/ 与 data/

mkdir -p ~/evalmatcher-output      # 结果输出目录（空目录即可，会自动创建）

# ③ 写 .env —— 只需要改三个路径
cp .env.example .env
sed -i \
  -e "s#^WEIGHTS_DIR=.*#WEIGHTS_DIR=$HOME/evalmatcher-weights#" \
  -e "s#^DATASET_DIR=.*#DATASET_DIR=$HOME/evalmatcher-data#" \
  -e "s#^OUTPUT_DIR=.*#OUTPUT_DIR=$HOME/evalmatcher-output#" .env

# ④ 一键启动（首次会构建镜像，torch 较大，约 10–20 分钟）
docker compose up -d --build

# ⑤ 看状态 / 日志
docker compose ps
docker compose logs -f server

# ⑥ 健康检查
curl -s http://127.0.0.1:8000/api/health          # 服务端
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/   # 前端，应返回 200
curl -s http://127.0.0.1:8080/api/config          # 经 nginx 反代拿到后端配置

# 浏览器打开
echo "http://$(hostname -I | awk '{print $1}'):8080/"
```

启动后第一次评测会加载模型（约 10–20 秒），之后就快了。

---

## 3. 配置怎么加载

**两处配置，各管一头，不混：**

| 文件 | 管什么 | 是否入库 |
|---|---|---|
| `.env` | 部署：端口、宿主机挂载路径、GPU 编号、配置文件路径、torch 版本 | ❌ 不入库（`.gitignore`） |
| `config/server.yaml` | 算法：方法清单、权重文件名、数据集、输出、device、阈值、最大关键点数 | ✅ 入库 |

加载顺序（后者覆盖前者）：

```
config/server.yaml
   └─► 环境变量覆盖：EM_WEIGHTS_DIR / EM_DATASET_DIR / EM_OUTPUT_DIR
                     EM_DEVICE / EM_PORT / EM_HOST
        （由 docker-compose 从 .env 注入，容器里是固定值 /app/weights 等）
             └─► config_loader.get()  ──►  app.py / eval_core.py
```

容器里的路径**永远是**这三个，服务端代码不认宿主机目录：

| 容器路径 | 来源 | 权限 |
|---|---|---|
| `/app/weights` | `.env: WEIGHTS_DIR` | 只读 `:ro` |
| `/app/data` | `.env: DATASET_DIR` | 只读 `:ro` |
| `/app/output` | `.env: OUTPUT_DIR` | 可写 `:rw` |
| `/app/config/server.yaml` | `.env: SERVER_CONFIG_FILE` | 只读 `:ro` |

第三方算法代码里写死了 `third/...` 的相对权重路径（例如 `third/MINIMA/weights/*.ckpt`），
为了**一行都不改算法**，启动时 `preflight.py` 会把 `/app/weights` 里的权重**软链**到这些位置。

---

## 4. 权重与数据集

两者都不在仓库、不进镜像（`*.pth / *.ckpt / *.pt` 已被 `.gitignore` 排除），
放在 Google Drive 上：

| 内容 | 链接 | 大小 |
|---|---|---|
| 数据集（VIS_SAR test） | https://drive.google.com/file/d/1DHfI1j4yELughX-ljwvNofgofMS8lHU2/view | 约 125 MB |
| 权重（5 个方法） | https://drive.google.com/file/d/1Y6D-Fu8-99f7buNCag1tH4C_JVO059Cy/view | 约 199 MB |

一键落盘（脚本里已经写好了这两个链接，直接跑）：

```bash
pip install gdown            # Google Drive 大文件有"病毒扫描确认页"，curl 直接下会拿到 HTML
./scripts/fetch_assets.sh    # -> 仓库下 weights/ 与 data/，并自动解压、抹平多余目录层级
```

也可以手动：浏览器打开上面两个链接下载，按下面的结构解压，再把绝对路径填进 `.env`。

### 权重目录（`WEIGHTS_DIR`）

```
<WEIGHTS_DIR>/
├── ours/
│   ├── 2024_10_10-10_44_34_VIS_SAR_106000.pth      # HLDD
│   ├── 2024_10_21-10_14_50_VIS_IR_62500.pth
│   └── 2024_11_02-09_43_00_VIS_NIR_30000.pth
├── redfeat/
│   ├── VIS_SAR.pth   VIS_IR.pth   VIS_NIR.pth
├── d2net/
│   └── d2_tf.pth
└── minima/
    ├── weights_xoftr_640.ckpt      # XoFTR
    └── minima_loftr.ckpt           # LoFTR
```

↑ 只需评测用到的 5 个方法，约 **199 MB**。文件名与 `config/server.yaml` 的
`methods[*].weights[].file` 一一对应；想换文件名，改 `server.yaml` 即可。

### 数据集目录（`DATASET_DIR`）

```
<DATASET_DIR>/
└── VIS_SAR/
    └── test/
        ├── VIS/          424 张 .png
        ├── SAR/          424 张同名 .png
        └── transforms/   424 个 .mat（如 1.png.12.mat / 1.png.21.mat，真值单应）
```

↑ 只需 `test/` 约 **125 MB**（`train/` 是训练用的，评测不需要）。

> 下载后脚本会打印目录检查；如果提示没找到 `VIS_SAR/test/{VIS,SAR,transforms}`，
> 说明压缩包多包了一层目录，手动把内容上移一层即可。

---

## 5. 常用命令

```bash
docker compose up -d --build       # 构建并启动
docker compose up -d               # 已构建过，直接启动
docker compose ps                  # 状态（healthy / unhealthy）
docker compose logs -f             # 全部日志
docker compose logs -f server      # 只看服务端
docker compose restart server      # 重启服务端
docker compose down                # 停止并移除容器
docker compose down -v             # 连卷一起删（挂载目录本身不动）

# 健康检查
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool
docker inspect --format '{{.State.Health.Status}}' evalmatcher-server
```

`/api/health` 会返回配置路径、启用方法、CUDA 是否可用、数据集/输出目录读写状态：

```json
{"status":"ok","methods":["d2net","redfeat","Ours","XoFTR","LoFTR"],
 "labels":{"Ours":"HLDD", ...},"cuda_available":true,"output_writable":true}
```

---

## 6. 服务端启动检查

容器启动时先跑 `preflight.py`，六项全过才起服务，**任何一项不过立即退出**并打印原因：

| 检查项 | 不过时会告诉你 |
|---|---|
| 配置文件 | 文件在哪、是 `SERVER_CONFIG` 没设还是 YAML 写错了 |
| 权重文件 | 缺哪几个文件、应该放到哪个目录 |
| 数据集目录 | 缺哪个子目录，并给出应有的目录结构 |
| 输出目录 | 不可写的原因（权限 / uid） |
| CUDA | `torch.cuda.is_available()` 为假时的三种排查方向 |
| 算法支持 | 方法 id 不在 `matcher.py` 支持列表里时，列出全部可用 id |

手动跑一次（不启动服务）：

```bash
docker compose run --rm server python preflight.py
```

---

## 7. 非 Docker 的开发启动方式

改造前的用法仍然保留：一个 Python 进程跑起来。

```bash
# 装依赖（建议用 conda/venv；torch 请按你的 CUDA 版本从 pytorch.org 装）
pip install -r server/requirements.txt

# 路径不再来自容器挂载，用环境变量指向本地目录
EM_WEIGHTS_DIR=~/evalmatcher-weights \
EM_DATASET_DIR=~/evalmatcher-data \
EM_OUTPUT_DIR=~/evalmatcher-output \
./scripts/run_dev.sh                       # 默认 8000 端口

PORT=8300 GPU=3 ./scripts/run_dev.sh       # 换端口 / 换卡
```

前端不能直接用 `file://` 打开（要 fetch `/api`），开发时可以用 client 容器单独起：

```bash
API_UPSTREAM=http://host.docker.internal:8000 docker compose up -d client
```

---

## 8. 常见错误排查

| 现象 | 原因 | 怎么处理 |
|---|---|---|
| `docker compose up` 报 `请在 .env 里设置 WEIGHTS_DIR` | 没写 `.env` 或路径没填 | `cp .env.example .env` 并填三个路径 |
| server 容器起来就退出，日志显示权重缺失 | 权重目录结构/文件名和 `server.yaml` 对不上 | 对照第 4 节，或跑 `docker compose run --rm server python preflight.py` 看缺哪个 |
| `CUDA 可用性 FAIL: torch.cuda.is_available() == False` | 没装 NVIDIA Container Toolkit，或 compose 的 GPU 段被删了 | 按第 1 节装 toolkit；确认 `docker-compose.yml` 的 `devices` 段还在；或把 `DEVICE=cpu` |
| 页面能开但方法列表是空的 | nginx 反代没到后端 | `curl http://127.0.0.1:8080/api/config`；看 `docker compose logs client` |
| `docker compose up --build` 在 pip 装 torch 那步很慢/失败 | 网络不通或 `CUDA_TAG` 与驱动不匹配 | 换 `--build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`；或改 `TORCH_VERSION/CUDA_TAG` |
| 评测跑到一半报显存不足 | 别的进程占着卡 | `nvidia-smi` 看占用；`.env` 换 `GPU_IDS`；或减少同时评测的方法 |
| 导出报表/连线图失败 | `OUTPUT_DIR` 不可写 | 确认宿主机目录存在且 docker 有权限；Linux 上注意目录 uid/gid |
| 端口被占 | `WEB_PORT` / `API_PORT` 与已有服务冲突 | 改 `.env` 里的端口 |

---

## 9. 目录结构

```
evalmatcher/
├── docker-compose.yml         一键编排（server + client，含 GPU 支持）
├── .env.example               部署配置模板（复制为 .env 后填路径）
├── config/server.yaml         算法配置（方法 / 权重 / 阈值 / device）
├── client/
│   ├── Dockerfile             nginx:alpine
│   ├── nginx.conf.template    /api 反代目标由 API_UPSTREAM 决定
│   ├── 40-inject-api-base.sh  可选：注入外部 API 地址
│   └── index.html             单页前端（零依赖，API 地址不写死）
├── server/
│   ├── Dockerfile             nvidia/cuda + torch(cu128)
│   ├── requirements.txt
│   ├── docker-entrypoint.sh   preflight -> 起服务
│   ├── config_loader.py       server.yaml + 环境变量
│   ├── preflight.py           六项启动检查 + 权重软链
│   ├── app.py                 Flask API
│   ├── eval_core.py           评测内核（指标口径与离线脚本一致）
│   ├── matcher.py / utils.py  方法分发 / 指标工具
│   └── third/                 第三方算法代码（d2net / redfeat / Ours / MINIMA）
└── scripts/
    ├── run_dev.sh             非 Docker 启动
    └── fetch_assets.sh        权重与数据集下载
```

关于方法名：内部注册名仍是 `Ours`（`matcher.py` 与 `third/Ours` 按它分发、存档文件名也是它），
对外显示 **HLDD**，映射在 `config/server.yaml` 的 `methods[*].label`，改一处即可。

---

## 10. 假设与说明

以下几点无法从原项目自动确定，按下面的假设处理，如与实际不符请改配置：

1. **torch 版本**：`2.11.0+cu128`（取自实际跑通的环境）。若你的驱动是其他 CUDA 版本，改 `.env` 的 `CUDA_TAG` 与 `TORCH_VERSION`。
2. **基础镜像**：默认 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` + pip 装 torch；若你本地已有 `pytorch/pytorch` 镜像，用 `--build-arg BASE_IMAGE=...` 可跳过 pip 装 torch。
3. **GPU 编号**：默认 `GPU_IDS=0`。多卡机器请按 `nvidia-smi` 的实际编号填写。
4. **数据集子集**：默认只有 `VIS_SAR`。加子集要同时改 `config/server.yaml` 的 `dataset.subsets` 与 `eval_core.py` 的权重映射。
5. **离线权威结果 `result/`** 不入库、不进镜像；连线图默认从**评测存档**现渲染（与逐对表格同源），有现成 PNG 时才走 `EM_OFFLINE_VIZ_DIR` 加速。
6. **评测任务表在进程内存里**，所以用多 worker 的 WSGI 会导致轮询找不到任务；如需 gunicorn 请固定 `--workers 1`。
