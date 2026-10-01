# -*- coding: utf-8 -*-
"""数据集级评测内核 —— `eval_model.py` 内层循环的逐行移植。

为什么要有这个文件
------------------
网页要"点击评测 -> 真跑 5 个方法 -> 出结果"，就必须能在**一次函数调用**里
算出与 `eval_model.py` **完全相同口径**的指标。所以这里不重新发明指标，
而是把 `eval_model.py` 的循环体原样搬过来，只额外加了：
  (a) 进度回调 on_event（前端要转圈）
  (b) 逐对记录 per-pair records（前端要显示每张图一行 + 支持任意子集聚合）
  (c) 可选的样本子集 pair_names（网页不可能一次跑完 424 对）

指标定义（与 `eval_model.py` 一字不改）
--------------------------------------
  MMA             : 每个阈值 t∈[1..10]，mean(dist <= t) 在全部图上取平均
  Homography      : DEGENSAC 预测 H 的四角平均位移 <= t 记为正确，在全部图上取均值
  Success Rate    : ncm>=10 的图数 / 图数
  Average Matches : mean(len(matches))
  NCM             : mean(ncm)，**仅**统计 ncm>=10 的图
  ME / RMSE       : 内点重投影误差的均值 / 均方根，**仅**统计 ncm>=10 的图
  Average Time    : mean(match_time[1:])  —— 丢掉第一张（含模型冷启），同原脚本

两个容易踩的"同源"细节
--------------------
1. 真值单应方向不固定：优先用 `<name>.12.mat`，**解析失败就整体退到** `.21.mat`
   并在调用 `cal_reproj_dists_H` 时把两组点交换。原脚本用裸 `except:` 实现，
   这里保持同样结构（连 `dist` 在两边都要重算这点也一致）。
2. `Average Time` 丢掉第一条耗时（`match_time[1:]`）；匹配抛异常的图**不**计入
   `match_time`（append 在 try 内、match_method 之后）。这两点都会直接影响
   与离线结果的比对，所以逐字保留。
"""
import os
import sys
import time

import numpy as np

WORKDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(WORKDIR)          # server/ 的上一层（开发机上 = 仓库根）
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# 数据集根目录：**从配置读**（容器里是 /app/data，由 .env 的 DATASET_DIR 只读挂载）。
# 拿不到配置就退回仓库内的 data/，这样离线脚本单独跑也不会炸。
def _data_root():
    try:
        import config_loader
        p = config_loader.dataset_dir()
        if os.path.isdir(p):
            return p
    except Exception:
        pass
    return os.path.join(REPO, 'data')


DATA_ROOT = _data_root()

# HLDD(Ours) 的权重路径同样从配置读；失败时退回 third/Ours/models 下的软链
# （preflight 会把 /app/weights 软链过去），两条路都通向同一份权重。
def _ours_weight(subset):
    try:
        import config_loader
        cfg = config_loader.get()
        wd = config_loader.weights_dir(cfg)
        for m in (cfg.get('methods') or []):
            if m.get('id') != 'Ours':
                continue
            for w in (m.get('weights') or []):
                if subset in os.path.basename(w.get('link', '')):
                    return os.path.join(wd, w['file'])
    except Exception:
        pass
    return {'VIS_SAR': 'third/Ours/models/2024_10_10-10_44_34_VIS_SAR_106000.pth',
            'VIS_IR': 'third/Ours/models/2024_10_21-10_14_50_VIS_IR_62500.pth',
            'VIS_NIR': 'third/Ours/models/2024_11_02-09_43_00_VIS_NIR_30000.pth'}[subset]

_THRES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
THRES_RANGE = list(range(1, 16))


def _cfg_eval(key, default):
    """从 config/server.yaml 的 eval 段读算法参数；读不到就用默认（离线脚本也能跑）。"""
    try:
        import config_loader
        return (config_loader.get().get('eval') or {}).get(key, default)
    except Exception:
        return default


SUCCESS_THRES = _cfg_eval('success_thres', 3)     # 误差阈值 3px
MIN_NCM = _cfg_eval('min_ncm', 10)                # ncm>=10 才算成功图
VIZ_MAX_WIDTH = _cfg_eval('viz_max_width', 1800)  # 连线图最大宽度

# MINIMA 的 load_model 会 argparse 解析 sys.argv；模块导入期就把它清干净，
# 否则网页请求带的参数会让它直接 exit。
_SAVED_ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]


def image_names(subset='VIS_SAR'):
    """返回该子集的图名列表，排序方式与 eval_model.py 一致（sorted）。"""
    p = os.path.join(DATA_ROOT, subset, 'test', subset.split('_')[0])
    return sorted(f for f in os.listdir(p) if f.lower().endswith('.png'))


def _load_gt(subset, name, suffix):
    import scipy.io as scio
    path = os.path.join(DATA_ROOT, subset, 'test', 'transforms',
                        name.replace('.png', suffix + '.mat'))
    return scio.loadmat(path)['H']


def _corner_dist(H_gt, H_pred, imgpath1, scale=1.0):
    """原脚本的 Homography 指标：真值角点 vs 预测角点的平均位移（像素）。"""
    from PIL import Image
    im = Image.open(imgpath1)
    w, h = im.size
    w, h = w / scale, h / scale
    corners = np.array([[0, 0, 1], [0, h - 1, 1], [w - 1, 0, 1], [w - 1, h - 1, 1]])
    real = np.dot(corners, np.transpose(H_gt))
    real = real[:, :2] / real[:, 2:]
    warped = np.dot(corners, np.transpose(H_pred))
    warped = warped[:, :2] / warped[:, 2:]
    return float(np.mean(np.linalg.norm(real - warped, axis=1)))


def evaluate(methods, subset='VIS_SAR', num_kps=1024, pair_names=None,
             matching=True, homography=True, ransac_thres=2, on_event=None,
             save_matches_dir=None):
    """跑一轮数据集评测。

    methods   : 方法名列表，如 ['d2net','redfeat','Ours','XoFTR','LoFTR']
    pair_names: 要评的图名子集；None = 全部
    on_event  : 回调 dict，事件类型见下面注释。前端靠它做实时进度。
    save_matches_dir: 若给出，每个方法跑完就把**逐对 matches** 存到
                      <dir>/<method>.npz（键=图像名）。这是连线图唯一可信的数据源：
                      连线图必须用"评测这一次的匹配"来画，不能事后重跑——
                      HLDD(XFeat) 的匹配在 CUDA 下跨进程会翻面（实测 29.png
                      评测时 ncm=4、重跑得 0），重跑出来的图和表格就对不上。

    返回 {'subset','num_kps','methods':{m: {'summary':..., 'pairs':[...]}}, ...}
    """
    import cv2  # noqa: F401  (matcher 内部依赖)
    import matcher
    from utils import cal_reproj_dists_H
    import pydegensac

    all_names = image_names(subset)
    names = list(pair_names) if pair_names else all_names
    unknown = [n for n in names if n not in set(all_names)]
    if unknown:
        raise ValueError('样本不存在: %s' % ','.join(unknown[:5]))

    filepath1 = os.path.join(DATA_ROOT, subset, 'test', subset.split('_')[0])
    filepath2 = os.path.join(DATA_ROOT, subset, 'test', subset.split('_')[1])

    def emit(kind, **kw):
        if on_event:
            try:
                on_event(dict(kind=kind, **kw))
            except Exception:
                pass

    out = {'subset': subset, 'num_kps': num_kps,
           'n_pairs': len(names), 'total_pairs': len(all_names),
           'methods': {}, 'method_order': list(methods)}

    for mi, method_name in enumerate(methods):
        emit('method_start', method=method_name, method_index=mi,
             n_methods=len(methods), n_pairs=len(names))

        xfeat = None
        if method_name == 'Ours':
            sys.path.append('third/Ours')
            from third.Ours.lib.xfeat import XFeat
            weights = _ours_weight(subset)
            xfeat = XFeat(weights=weights)

        match_failed = 0
        n_matches, match_time = [], []
        thre_err = {t: 0.0 for t in THRES_RANGE}
        dists_sa, inlier_ratio = [], []
        all_NCM, all_ME, all_RMSE, all_failed = [], [], [], []
        SR = 0
        pairs = []
        matches_cache = {}          # 逐对 matches 存档（连线图的数据源）

        for idx, name in enumerate(names):
            imgpath1 = os.path.join(filepath1, name)
            imgpath2 = os.path.join(filepath2, name)
            rec = {'name': name, 'error': None, 'suffix': None}

            # ---------- 匹配（与 eval_model.py 完全同构） ----------
            try:
                t0 = time.time()
                matches, kp1, kp2 = matcher.match_method(
                    method_name, imgpath1, imgpath2, num_kps, subset, xfeat)
                dt = time.time() - t0
                match_time.append(dt)
                if not isinstance(matches, np.ndarray):
                    matches = matches.cpu().numpy()
            except Exception as e:
                dt = None
                p1s = p2s = matches = []
                match_failed += 1
                rec['error'] = '%s: %s' % (type(e).__name__, e)
            rec['match_time'] = dt
            n_matches.append(len(matches))
            rec['n_matches'] = int(len(matches))
            if save_matches_dir is not None:
                matches_cache[name] = np.asarray(matches, dtype=np.float32)

            H_pred, inliers = None, []
            # ---------- 真值单应：先 .12，失败整体退 .21 ----------
            try:
                suffix = '.12'
                H_gt = _load_gt(subset, name, suffix)
                if matching:
                    dist = (np.array([float('inf')]) if len(matches) == 0
                            else cal_reproj_dists_H(matches[:, 2:], matches[:, :2], H_gt))
                    for thr in THRES_RANGE:
                        thre_err[thr] += np.mean(dist <= thr)
                if homography:
                    try:
                        H_pred, inliers = pydegensac.findHomography(
                            matches[:, 2:], matches[:, :2], ransac_thres)
                    except Exception:
                        H_pred = None
                    if H_pred is None:
                        corner_dist, h_failed, irat = np.nan, 1, 0
                        inliers = []
                    else:
                        corner_dist = _corner_dist(H_gt, H_pred, imgpath1)
                        irat = np.mean(inliers)
            except Exception:
                suffix = '.21'
                H_gt = _load_gt(subset, name, suffix)
                if matching:
                    dist = (np.array([float('inf')]) if len(matches) == 0
                            else cal_reproj_dists_H(matches[:, :2], matches[:, 2:], H_gt))
                    for thr in THRES_RANGE:
                        thre_err[thr] += np.mean(dist <= thr)
                if homography:
                    try:
                        H_pred, inliers = pydegensac.findHomography(
                            matches[:, :2], matches[:, 2:], ransac_thres)
                    except Exception:
                        H_pred = None
                    if H_pred is None:
                        corner_dist, h_failed, irat = np.nan, 1, 0
                        inliers = []
                    else:
                        corner_dist = _corner_dist(H_gt, H_pred, imgpath1)
                        irat = np.mean(inliers)

            # ---------- 逐对指标 ----------
            mask_th_3 = dist <= SUCCESS_THRES
            ncm = int(np.sum(mask_th_3))
            rec.update({'suffix': suffix, 'ncm': ncm,
                        # 存下来是为了支持"只对某个子集重新聚合"时口径完全一致
                        'mma_contrib': [float(np.mean(dist <= t)) for t in _THRES],
                        'corner_dist': (None if np.isnan(corner_dist) else float(corner_dist)),
                        'inlier_ratio': float(np.mean(irat)) if hasattr(irat, 'shape') else float(irat)})
            if ncm >= MIN_NCM:
                SR += 1
                all_NCM.append(ncm)
                me = float(np.sum(dist[mask_th_3]) / ncm)
                rmse = float(np.sqrt(np.sum(dist[mask_th_3] ** 2) / ncm))
                all_ME.append(me)
                all_RMSE.append(rmse)
                rec.update({'me': me, 'rmse': rmse, 'success': True})
            else:
                all_failed.append(name)
                rec.update({'me': None, 'rmse': None, 'success': False})

            inlier_ratio.append(irat)
            dists_sa.append(corner_dist)
            pairs.append(rec)
            emit('pair_done', method=method_name, method_index=mi, index=idx,
                 name=name, record=rec, n_pairs=len(names))

        # ---------- 汇总（与原脚本同一套公式） ----------
        n_img = len(names)
        mma = [thre_err[t] / n_img for t in _THRES]
        homography_curve = np.mean(
            [[float(d <= t) for t in _THRES] for d in dists_sa], axis=0).tolist()
        summary = {
            'method': method_name,
            'n_pairs': n_img,
            'MMA': [float(v) for v in mma],
            'Homography': [float(v) for v in homography_curve],
            'SuccessRate': float(SR / n_img),
            'AverageMatches': float(np.mean(n_matches)),
            'NCM': float(np.mean(all_NCM)) if all_NCM else 0.0,
            'ME': float(np.mean(all_ME)) if all_ME else None,
            'RMSE': float(np.mean(all_RMSE)) if all_RMSE else None,
            'AverageTime': (float(np.mean(match_time[1:])) if len(match_time) > 1
                            else (float(match_time[0]) if match_time else None)),
            'failed_count': len(all_failed),
            'failed_images': all_failed,
            'match_failed': match_failed,
            'n_success_pairs': len(all_NCM),
        }
        out['methods'][method_name] = {'summary': summary, 'pairs': pairs}
        if save_matches_dir is not None:
            os.makedirs(save_matches_dir, exist_ok=True)
            np.savez(os.path.join(save_matches_dir, '%s.npz' % method_name),
                     **matches_cache)
        emit('method_done', method=method_name, method_index=mi,
             summary=summary, n_methods=len(methods))

    return out


def _ours_xfeat(subset):
    sys.path.append('third/Ours')
    from third.Ours.lib.xfeat import XFeat
    return XFeat(weights=_ours_weight(subset))


def render_matches(method, subset, name, num_kps=1024, max_width=None,
                   matches_dir=None, job_label=None, info=None):
    """画某方法在某对图上的匹配连线（左右拼图 + 连线），返回 PNG bytes。

    连线配色与 `utils.visual_matching` 同一口径：到真值单应的重投影误差 <=3px 画绿，
    否则画红。所以这张图本身就是"哪些匹配是对的"的证据，而不是装饰。

    **数据源是这张图的命门**：matches_dir 给出且能读到存档时，用**评测那一次**存下来的
    matches 画（src='archive'，与逐对表格同源，逐对必然对得上）；读不到才现场重跑
    （src='rerun'）。重跑对多数方法是确定性的，但 HLDD(XFeat) 跨进程会翻面
    ——29.png 评测时 ncm=4、重跑得 0——重跑画出来的图会和表格对不上，所以**必须**在图上
    标出来，别让人拿它当评测结果。
    """
    import io
    import matcher
    from utils import cal_reproj_dists_H
    from PIL import Image, ImageDraw

    filepath1 = os.path.join(DATA_ROOT, subset, 'test', subset.split('_')[0])
    filepath2 = os.path.join(DATA_ROOT, subset, 'test', subset.split('_')[1])
    p1 = os.path.join(filepath1, name)
    p2 = os.path.join(filepath2, name)

    xfeat = None
    src = 'rerun'
    matches = None
    if matches_dir:
        npz = os.path.join(matches_dir, '%s.npz' % method)
        if os.path.exists(npz):
            try:
                with np.load(npz) as _d:
                    if name in _d.files:
                        matches = np.asarray(_d[name], dtype=float)
                        src = 'archive'
            except Exception:
                matches = None
    if src == 'rerun':
        xfeat = _ours_xfeat(subset) if method == 'Ours' else None
        matches, _, _ = matcher.match_method(method, p1, p2, num_kps, subset, xfeat)
    if not isinstance(matches, np.ndarray):
        matches = matches.cpu().numpy()
    matches = np.asarray(matches, dtype=float).reshape(-1, 4) if len(matches) else np.zeros((0, 4))

    # 真值单应：与主循环同一套"先 .12 失败退 .21"的规则，只为给连线上色
    dist = None
    for suffix in ('.12', '.21'):
        try:
            H_gt = _load_gt(subset, name, suffix)
            if len(matches):
                dist = (cal_reproj_dists_H(matches[:, 2:], matches[:, :2], H_gt)
                        if suffix == '.12'
                        else cal_reproj_dists_H(matches[:, :2], matches[:, 2:], H_gt))
            else:
                dist = np.zeros(0)
            break
        except Exception:
            continue

    im1 = Image.open(p1).convert('RGB')
    im2 = Image.open(p2).convert('RGB')
    canvas = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)),
                       (16, 16, 20))
    canvas.paste(im1, (0, 0))
    canvas.paste(im2, (im1.width, 0))
    d = ImageDraw.Draw(canvas)
    off = im1.width
    for i in range(len(matches)):
        x1, y1, x2, y2 = matches[i]
        ok = dist is not None and i < len(dist) and dist[i] <= SUCCESS_THRES
        d.line([float(x1), float(y1), float(x2) + off, float(y2)],
               fill=(0, 230, 118) if ok else (255, 82, 82), width=1)
    # 数据源标注：存档=与表格同源；重跑=可能与表格不一致，必须画出来提醒
    note = ('ARCHIVE %s' % (job_label or '')) if src == 'archive' \
        else 'RERUN (not from eval archive)'
    d.rectangle([canvas.width - 220, canvas.height - 22, canvas.width - 4, canvas.height - 4],
                fill=(0, 0, 0))
    d.text((canvas.width - 212, canvas.height - 16), note,
           fill=(0, 230, 118) if src == 'archive' else (255, 180, 60))
    max_width = max_width or VIZ_MAX_WIDTH
    if canvas.width > max_width:
        r = max_width / canvas.width
        canvas = canvas.resize((max_width, max(1, int(canvas.height * r))), Image.LANCZOS)
    buf = io.BytesIO()
    canvas.save(buf, 'PNG', optimize=True)
    if info is not None:
        info.update({'source': src, 'n_matches': int(len(matches)),
                     'ncm': int(np.sum(dist <= SUCCESS_THRES)) if dist is not None else None})
    return buf.getvalue()


def aggregate(pairs):
    """从逐对记录重新聚合出 summary —— 前端按子集显示时用同一套公式。"""
    n_img = len(pairs) or 1
    mma = np.zeros(len(_THRES))
    hom = []
    for r in pairs:
        d = r.get('mma_contrib')
        if d:
            mma += np.array(d)
        cd = r.get('corner_dist')
        hom.append([float(0.0) for _ in _THRES] if cd is None
                   else [float(cd <= t) for t in _THRES])
    succ = [r for r in pairs if r.get('success')]
    times = [r['match_time'] for r in pairs if r.get('match_time') is not None]
    return {
        'n_pairs': len(pairs),
        'MMA': (mma / n_img).tolist(),
        'Homography': np.mean(hom, axis=0).tolist() if hom else [],
        'SuccessRate': len(succ) / n_img,
        'AverageMatches': float(np.mean([r['n_matches'] for r in pairs])) if pairs else 0.0,
        'NCM': float(np.mean([r['ncm'] for r in succ])) if succ else 0.0,
        'ME': float(np.mean([r['me'] for r in succ])) if succ else None,
        'RMSE': float(np.mean([r['rmse'] for r in succ])) if succ else None,
        'AverageTime': (float(np.mean(times[1:])) if len(times) > 1
                        else (float(times[0]) if times else None)),
        'failed_count': sum(1 for r in pairs if not r.get('success')),
    }
