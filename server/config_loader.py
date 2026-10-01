# -*- coding: utf-8 -*-
"""配置加载：config/server.yaml + 环境变量覆盖。

职责边界
--------
* 本模块**只管配置**，不管算法。算法仍然是 matcher.py / eval_core.py 那套。
* 路径一律是**容器内部固定路径**（/app/weights、/app/data、/app/output），
  开发机上想指向别处用环境变量覆盖，不要改 server.yaml。

加载顺序（后者覆盖前者）
------------------------
1. config/server.yaml
2. 环境变量 EM_WEIGHTS_DIR / EM_DATASET_DIR / EM_OUTPUT_DIR / EM_DEVICE / EM_PORT
3. （仅路径相关）.env 里的挂载点由 docker-compose 传进来，最终落到同一批环境变量

为什么允许 env 覆盖路径
----------------------
挂载点是部署时（.env）才确定的，而 server.yaml 是仓库里的算法配置。
部署者改了 .env 的宿主机目录却不改 server.yaml 也能跑 —— 只要容器内的
挂载点不变；连挂载点都想改，就用 EM_* 覆盖。
"""
import os
import sys

try:
    import yaml
except ImportError:  # 容器里 requirements.txt 已装；这里是给开发机一个明确报错
    sys.stderr.write('[config] 缺少 PyYAML，请先 pip install pyyaml\n')
    raise

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))


class ConfigError(Exception):
    """配置缺失或非法。启动检查会把它转成清晰的错误信息并退出。"""


def _default_config_path():
    # 1) 显式指定
    p = os.environ.get('SERVER_CONFIG') or os.environ.get('EM_CONFIG')
    if p:
        return os.path.abspath(p)
    # 2) 容器里的固定位置
    if os.path.isfile('/app/config/server.yaml'):
        return '/app/config/server.yaml'
    # 3) 开发模式：仓库里的 config/server.yaml
    p = os.path.join(os.path.dirname(SERVER_DIR), 'config', 'server.yaml')
    return p


def _deep_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config(path=None):
    """读 YAML 并做 env 覆盖，返回一个普通 dict。不做存在性校验（那是 preflight 的事）。"""
    path = path or _default_config_path()
    if not os.path.isfile(path):
        raise ConfigError(
            '配置文件不存在：%s\n'
            '  容器内默认 /app/config/server.yaml（由 docker-compose 挂载）；\n'
            '  也可以用 SERVER_CONFIG=/path/to/server.yaml 指定。' % path)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise ConfigError('配置文件不是合法 YAML：%s\n  %s: %s' % (path, type(e).__name__, e))
    if not isinstance(cfg, dict):
        raise ConfigError('配置文件内容不是映射（dict）：%s' % path)

    paths = dict(cfg.get('paths') or {})
    # ---- 环境变量覆盖（部署侧参数）----
    env_map = {
        'EM_WEIGHTS_DIR': 'weights_dir',
        'EM_DATASET_DIR': 'dataset_dir',
        'EM_OUTPUT_DIR': 'output_dir',
        'EM_APP_ROOT': 'app_root',
    }
    for ek, key in env_map.items():
        v = os.environ.get(ek)
        if v:
            paths[key] = v
    cfg['paths'] = paths

    server = dict(cfg.get('server') or {})
    if os.environ.get('EM_PORT'):
        server['port'] = int(os.environ['EM_PORT'])
    if os.environ.get('EM_HOST'):
        server['host'] = os.environ['EM_HOST']
    if os.environ.get('EM_DEVICE'):
        server['device'] = os.environ['EM_DEVICE']
    cfg['server'] = server

    cfg['_config_path'] = path
    return cfg


_CFG = None


def get(reload=False):
    """进程内单例。"""
    global _CFG
    if _CFG is None or reload:
        _CFG = load_config()
    return _CFG


def weights_dir(cfg=None):
    return (cfg or get())['paths']['weights_dir']


def dataset_dir(cfg=None):
    return (cfg or get())['paths']['dataset_dir']


def output_dir(cfg=None):
    return (cfg or get())['paths']['output_dir']


def app_root(cfg=None):
    return (cfg or get())['paths'].get('app_root') or SERVER_DIR


def enabled_methods(cfg=None):
    return [m for m in ((cfg or get()).get('methods') or []) if m.get('enabled', True)]


def method_ids(cfg=None):
    return [m['id'] for m in enabled_methods(cfg)]


def labels(cfg=None):
    """内部 id -> 对外显示名（页面/报表/日志都走这个映射）。"""
    return {m['id']: m.get('label', m['id']) for m in enabled_methods(cfg)}


def weight_links(cfg=None):
    """[(宿主机权重文件绝对路径, 容器内算法期望路径)]，用于启动时建软链。"""
    cfg = cfg or get()
    wd = weights_dir(cfg)
    root = app_root(cfg)
    out = []
    for m in enabled_methods(cfg):
        for w in (m.get('weights') or []):
            src = os.path.join(wd, w['file'])
            dst = os.path.join(root, w['link'])
            out.append((src, dst))
    return out
