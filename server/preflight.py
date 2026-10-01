# -*- coding: utf-8 -*-
"""启动检查（preflight）：配置 / 权重 / 数据集 / 输出 / CUDA / 算法 六项。

设计原则：**失败就立刻退出，并且把"缺什么、该做什么"说清楚**。
半死不活地起来、等到用户点了"开始评测"才报 FileNotFoundError 是最糟的体验。

用法
----
    python preflight.py                    # 检查 + 建权重软链，失败 exit(1)
    python preflight.py --no-link          # 只检查，不建软链
    python preflight.py --json             # 机器可读输出
"""
import argparse
import json
import os
import sys

import config_loader as C


# matcher.py 里 if 链上真实存在的分支（改 matcher.py 时同步这里）
SUPPORTED_METHOD_IDS = {
    'xfeat', 'd2net', 'r2d2', 'superglue', 'loftr', 'dedode', 'subpixel', 'alike',
    'redfeat', 'sift', 'Ours', 'SRIF', 'POS-GIFT', 'omniglue', 'subpx',
    'MINIMA_LG', 'MINIMA_LoFTR', 'MINIMA_RoMa', 'LoFTR', 'RoMa', 'MINIMA_XoFTR',
    'XoFTR', 'LightGlue',
}


class Check(object):
    def __init__(self, name, ok, detail, fix=''):
        self.name, self.ok, self.detail, self.fix = name, ok, detail, fix

    def as_dict(self):
        return {'name': self.name, 'ok': self.ok, 'detail': self.detail, 'fix': self.fix}


def _check_config():
    try:
        cfg = C.get()
    except C.ConfigError as e:
        return None, Check('配置文件', False, str(e),
                           '确认 docker-compose 把 config/server.yaml 挂进了容器，'
                           '或用 SERVER_CONFIG 指定路径')
    return cfg, Check('配置文件', True, cfg['_config_path'])


def _check_weights(cfg):
    links = C.weight_links(cfg)
    if not links:
        return Check('权重文件', False, 'server.yaml 的 methods[*].weights 为空',
                     '至少配置一个方法的权重')
    missing = [src for src, _ in links if not os.path.isfile(src)]
    if missing:
        return Check('权重文件', False,
                     '缺少 %d/%d 个权重文件，例如：\n      %s'
                     % (len(missing), len(links), '\n      '.join(missing[:5])),
                     '把权重放到 .env 的 WEIGHTS_DIR 指向的目录（容器里是 %s），'
                     '目录结构与 server.yaml 的 file 字段对应' % C.weights_dir(cfg))
    return Check('权重文件', True, '%d 个权重文件齐全（%s）' % (len(links), C.weights_dir(cfg)))


def _check_dataset(cfg):
    root = C.dataset_dir(cfg)
    if not os.path.isdir(root):
        return Check('数据集目录', False, '目录不存在：%s' % root,
                     '把数据集放到 .env 的 DATASET_DIR 指向的目录（容器里是 %s）' % root)
    subs = (cfg.get('dataset') or {}).get('subsets') or ['VIS_SAR']
    bad = []
    detail = []
    for s in subs:
        a, b = s.split('_')[0], s.split('_')[1]
        for part in (a, b, 'transforms'):
            p = os.path.join(root, s, 'test', part)
            if not os.path.isdir(p):
                bad.append(p)
        n = len([f for f in os.listdir(os.path.join(root, s, 'test', a))
                 if f.lower().endswith('.png')]) if not bad else 0
        detail.append('%s: %d 对' % (s, n))
    if bad:
        return Check('数据集目录', False,
                     '缺少子目录：\n      %s' % '\n      '.join(bad[:5]),
                     '数据集结构应为 <DATASET_DIR>/<SUBSET>/test/{<VIS>,<SAR>,transforms}')
    return Check('数据集目录', True, '%s（%s）' % (root, ', '.join(detail)))


def _check_output(cfg):
    root = C.output_dir(cfg)
    try:
        os.makedirs(root, exist_ok=True)
    except Exception as e:
        return Check('输出目录', False, '无法创建 %s：%s' % (root, e),
                     '确认宿主机目录存在且 docker 有权限写（Linux 上注意 uid/gid）')
    probe = os.path.join(root, '.write_test')
    try:
        with open(probe, 'w') as f:
            f.write('ok')
        os.remove(probe)
    except Exception as e:
        return Check('输出目录', False, '目录不可写：%s（%s）' % (root, e),
                     'chmod 该目录，或在 .env 里换一个 OUTPUT_DIR')
    return Check('输出目录', True, '可写：%s' % root)


def _check_device(cfg):
    dev = (cfg.get('server') or {}).get('device', 'cuda')
    if dev != 'cuda':
        return Check('CUDA 可用性', True, 'device=%s（按配置走 CPU）' % dev)
    try:
        import torch
    except Exception as e:
        return Check('CUDA 可用性', False, '无法 import torch：%s' % e,
                     '检查 server 镜像依赖是否装全')
    if not torch.cuda.is_available():
        return Check('CUDA 可用性', False,
                     'torch.cuda.is_available() == False',
                     '1) 宿主机装 NVIDIA 驱动 + NVIDIA Container Toolkit；'
                     '2) docker-compose.yml 的 deploy.resources.reservations.devices 保留；'
                     '3) 只想跑 CPU 冒烟就把 server.yaml 的 device 改成 cpu')
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    return Check('CUDA 可用性', True, 'device_count=%d, 当前 %s' % (n, name))


def _check_methods(cfg):
    ms = cfg.get('methods') or []
    if not ms:
        return Check('算法支持', False, 'server.yaml 的 methods 为空', '至少启用一个方法')
    ids = [m.get('id') for m in ms]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        return Check('算法支持', False, '方法 id 重复：%s' % dup, 'server.yaml 里去掉重复项')
    bad = [i for i in ids if i not in SUPPORTED_METHOD_IDS]
    if bad:
        return Check('算法支持', False,
                     '不受支持的方法 id：%s' % bad,
                     '可用 id：%s' % ', '.join(sorted(SUPPORTED_METHOD_IDS)))
    return Check('算法支持', True, '%d 个方法：%s' % (len(ids), ', '.join(ids)))


def ensure_weight_links(cfg, verbose=True):
    """把 /app/weights 下的权重软链到算法代码写死的 third/... 位置。

    这样算法代码一行都不用改 —— 它以为权重就在仓库里，实际在读只读挂载。
    若目标位置已经是**真实文件**（开发机上权重就在位），不动它，避免误删。
    """
    linked, skipped = 0, 0
    for src, dst in C.weight_links(cfg):
        d = os.path.dirname(dst)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        if os.path.islink(dst):
            if os.readlink(dst) == src:
                skipped += 1
                continue
            os.remove(dst)
        elif os.path.exists(dst):
            skipped += 1          # 真实权重已在位（非容器部署），不覆盖
            continue
        os.symlink(src, dst)
        linked += 1
    if verbose:
        print('[preflight] 权重软链：新建 %d，已在位 %d' % (linked, skipped))
    return linked, skipped


def run_all(make_links=True, verbose=True):
    cfg, chk_cfg = _check_config()
    checks = [chk_cfg]
    if cfg is None:
        return False, checks
    checks += [_check_weights(cfg), _check_dataset(cfg), _check_output(cfg),
               _check_device(cfg), _check_methods(cfg)]
    ok = all(c.ok for c in checks)
    if ok and make_links:
        try:
            ensure_weight_links(cfg, verbose=verbose)
        except Exception as e:
            ok = False
            checks.append(Check('权重软链', False, '%s: %s' % (type(e).__name__, e),
                                '检查 /app/weights 是否为只读挂载'))
    return ok, checks


def main():
    ap = argparse.ArgumentParser(description='EvalMatcher 服务端启动检查')
    ap.add_argument('--no-link', action='store_true', help='只检查，不建权重软链')
    ap.add_argument('--json', action='store_true', help='输出 JSON')
    a = ap.parse_args()

    ok, checks = run_all(make_links=not a.no_link, verbose=not a.json)
    if a.json:
        print(json.dumps({'ok': ok, 'checks': [c.as_dict() for c in checks]},
                         ensure_ascii=False, indent=2))
        return 0 if ok else 1

    print('=' * 66)
    print('EvalMatcher 启动检查')
    print('=' * 66)
    for c in checks:
        print('[%s] %-10s %s' % ('OK ' if c.ok else 'FAIL', c.name, c.detail))
        if not c.ok and c.fix:
            print('       -> 怎么做：%s' % c.fix)
    print('-' * 66)
    if ok:
        print('全部通过，启动服务…')
        return 0
    print('启动检查未通过，已终止。修完上面几项再启动。')
    return 1


if __name__ == '__main__':
    sys.exit(main())
