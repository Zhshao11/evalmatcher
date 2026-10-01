# -*- coding: utf-8 -*-
"""VIS_SAR 数据集评测 · 在线实时评测服务（跑在 GPU 机器上，只负责暴露端口）。

页面定位
--------
不是"把跑好的结果贴出来"，而是**点击「开始评测」-> 后端真跑这几个方法 ->
逐对回传进度与指标 -> 前端边跑边显示**。所以必须有真后端。

接口
----
  GET  /              评测页
  GET  /api/config    方法 / 子集 / 阈值 / 总对数
  GET  /api/pairs     图名列表（"单对"下拉用）
  GET  /api/image     数据集原图（side=vis|sar）
  POST /api/evaluate  提交评测 -> 立刻返回 job_id（后台线程跑）
  GET  /api/job       轮询进度 + 各方法汇总（不含逐对明细，包很小）
  GET  /api/viz       按需渲染某方法在某对图上的匹配连线
  GET  /api/exports           导出物清单（xlsx / 5 个方法连线图包）：是否存在、体积、构建进度
  GET  /api/export/xlsx       逐对明细导出报表（3 个 sheet 的 .xlsx）
  GET  /api/export/viz        某方法的 424 张匹配连线图打包（JPEG zip）
  GET  /api/evidence/gpu      GPU 运行证据（真实执行 nvidia-smi + torch 设备信息）
  GET  /api/evidence/jobs     已落盘的评测任务存档列表

为什么异步
---------
单请求超过 60s 会被上层网关掐断，而一次数据集评测远超 60s。
所以"提交 + 轮询"，每个 HTTP 请求都很短。

硬约束
------
* **不写盘到 result/**。result/ 是离线全量结果的权威副本，网页重跑覆盖它会让
  "官方口径"的证据链断掉。本文件新增的导出与取证一律写在 **web/evidence/** 下，
  与 result/ 完全隔离。
* 同时只跑一个评测任务（显存 + 结果可比性）。
"""
import io
import os
import sys
import time
import json
import glob
import uuid
import socket
import zipfile
import threading
import subprocess
import traceback
from pathlib import Path
from urllib.parse import quote

from flask import Flask, request, jsonify, send_from_directory, send_file, Response

# ---- 配置驱动的路径与算法清单 -------------------------------------------
# 容器里 server/ 就是 /app：third/、data/、weights/ 全都挂在它下面。
# 开发机上用 EM_WEIGHTS_DIR / EM_DATASET_DIR / EM_OUTPUT_DIR 覆盖，别改代码。
import config_loader  # noqa: E402

try:
    _CFG = config_loader.get()
except config_loader.ConfigError as e:
    sys.stderr.write('[app] 配置加载失败，服务无法启动：\n%s\n' % e)
    sys.stderr.write('[app] 先跑一次：python preflight.py  看具体缺什么\n')
    sys.exit(1)

WORKDIR = Path(__file__).parent
APP_ROOT = Path(config_loader.app_root(_CFG))     # 容器里 = /app
sys.path.insert(0, str(WORKDIR))
sys.path.insert(0, str(APP_ROOT))
os.chdir(str(APP_ROOT))              # matcher / eval_core 大量用相对路径（third/...）

import eval_core  # noqa: E402

app = Flask(__name__, static_folder=None)

SUBSETS = (_CFG.get('dataset') or {}).get('subsets') or ['VIS_SAR']
_EV = _CFG.get('eval') or {}
NUM_KPS_CHOICES = _EV.get('num_kps_choices') or [1024, 2048, 4096]
METHODS_ALL = config_loader.method_ids(_CFG)

# 内部 id 与对外显示名分离（映射来自 config/server.yaml 的 methods[*].label）。
#   'Ours' 是 matcher.py 里的注册名：third/Ours 目录、方法分发、评测存档的
#   npz 文件名（Ours.npz）、/api/viz?method=Ours 全都按它走，不能改；
#   对外显示名在 server.yaml 里配成 HLDD —— 不管实现是什么，名字就叫 HLDD。
LABELS = config_loader.labels(_CFG)
NAME2ID = {v: k for k, v in LABELS.items()}


def disp_name(m):
    """内部 id -> 对外显示名。页面上不该再出现 'Ours'。"""
    return LABELS.get(m, m)

_LOCK = threading.Lock()
_JOBS = {}
_ACTIVE = {'id': None}
_JOBS_MAX = 6


def _evict():
    with _LOCK:
        if len(_JOBS) <= _JOBS_MAX:
            return
        for k in sorted(_JOBS, key=lambda x: _JOBS[x]['created']):
            if len(_JOBS) <= _JOBS_MAX:
                break
            if _JOBS[k]['status'] == 'running':
                continue
            _JOBS.pop(k, None)


def _snapshot(job, pairs_for=None):
    with _LOCK:
        snap = {
            'id': job['id'], 'status': job['status'], 'error': job['error'],
            'subset': job['subset'], 'num_kps': job['num_kps'],
            'methods': job['methods'], 'n_pairs': job['n_pairs'],
            'total_pairs_in_subset': job['total_pairs_in_subset'],
            'scope_desc': job['scope_desc'], 'scope': job.get('scope'),
            # 任务结束后 finished 已落定，耗时冻结；running 阶段才实时累加。
            'elapsed': round((job.get('finished') or time.time()) - job['created'], 1),
            'progress': dict(job['progress']),
            'summaries': {m: (job['results'][m]['summary']
                              if job['results'].get(m) else None)
                          for m in job['methods']},
            'n_done_pairs': {m: len(job['results'][m]['pairs'])
                             for m in job['methods'] if job['results'].get(m)},
            'events': job['events'][-60:],
        }
        if pairs_for and job['results'].get(pairs_for):
            snap['pairs'] = list(job['results'][pairs_for]['pairs'])
            snap['pairs_for'] = pairs_for
        return snap


def _resolve_scope(payload):
    subset = payload.get('subset', 'VIS_SAR')
    if subset not in SUBSETS:
        raise ValueError('不支持的子集: %s' % subset)
    all_names = eval_core.image_names(subset)
    scope = payload.get('scope') or {'mode': 'first', 'n': 20}
    mode = scope.get('mode', 'first')

    if mode == 'all':
        names, desc = all_names, '全部 %d 对' % len(all_names)
    elif mode == 'single':
        name = scope.get('name')
        if name not in set(all_names):
            raise ValueError('样本不存在: %s' % name)
        names, desc = [name], '单对 %s' % name
    elif mode == 'list':
        names = [n for n in (scope.get('names') or []) if n in set(all_names)]
        if not names:
            raise ValueError('自定义样本列表为空')
        desc = '自定义 %d 对' % len(names)
    elif mode == 'range':
        a, b = int(scope.get('start', 0)), int(scope.get('end', 0))
        a = max(0, min(a, len(all_names)))
        b = max(a + 1, min(b, len(all_names)))
        names, desc = all_names[a:b], '第 %d~%d 对' % (a + 1, b)
    else:
        n = min(max(1, int(scope.get('n', 20))), len(all_names))
        names, desc = all_names[:n], '前 %d 对' % n

    return subset, names, desc, len(all_names)


def _start_job(payload):
    raw_scope = payload.get('scope') or {'mode': 'first', 'n': 20}
    subset, names, desc, total = _resolve_scope(payload)
    methods = [m for m in (payload.get('methods') or METHODS_ALL) if m in METHODS_ALL]
    if not methods:
        raise ValueError('至少要选一个方法')
    num_kps = int(payload.get('num_kps') or 1024)
    if num_kps not in NUM_KPS_CHOICES:
        num_kps = 1024

    job_id = uuid.uuid4().hex[:12]
    job = {
        'id': job_id, 'status': 'running', 'error': None, 'created': time.time(),
        'subset': subset, 'num_kps': num_kps, 'methods': methods,
        'n_pairs': len(names), 'total_pairs_in_subset': total, 'scope_desc': desc,
        'scope': dict(raw_scope),
        'progress': {'method': None, 'method_index': -1, 'n_methods': len(methods),
                     'pair_index': -1, 'pair_name': None, 'pairs_done': 0,
                     'pairs_total': len(names) * len(methods), 'phase': 'pending'},
        'results': {}, 'events': [],
    }
    with _LOCK:
        _JOBS[job_id] = job
        _ACTIVE['id'] = job_id
    _evict()
    _save_gpu_evidence(job, 'start')          # 任务开跑时的机器状态（取证用）

    def _worker():
        try:
            for mi, method in enumerate(methods):
                with _LOCK:
                    job['results'][method] = {'summary': None, 'pairs': []}
                    job['progress'].update({'method': method, 'method_index': mi,
                                            'pair_index': -1, 'pair_name': None,
                                            'phase': 'running'})
                    job['events'].append('▶ %s 开始（%d 对）' % (disp_name(method), len(names)))

                def on_event(ev, _m=method, _mi=mi):
                    with _LOCK:
                        k = ev.get('kind')
                        if k == 'pair_done':
                            rec = dict(ev['record'])
                            job['results'][_m]['pairs'].append(rec)
                            job['progress'].update({
                                'pair_index': ev['index'], 'pair_name': ev['name'],
                                'pairs_done': job['progress']['pairs_done'] + 1})
                            if ev['index'] == 0:
                                job['events'].append('  首对 %s: %d 匹配, ncm=%d%s' % (
                                    ev['name'], rec['n_matches'], rec['ncm'],
                                    (' , %.2fs' % rec['match_time'])
                                    if rec.get('match_time') else ' , 匹配异常'))
                        elif k == 'method_done':
                            s = ev['summary']
                            job['events'].append('✔ %s 完成  SR=%.4f  NCM=%.2f  AM=%.1f' % (
                                disp_name(_m), s['SuccessRate'], s['NCM'], s['AverageMatches']))

                out = eval_core.evaluate(
                    [method], subset=subset, num_kps=num_kps,
                    pair_names=names, on_event=on_event,
                    save_matches_dir=str(_matches_dir(job_id)))
                with _LOCK:
                    job['results'][method]['summary'] = out['methods'][method]['summary']
            with _LOCK:
                job['status'] = 'done'
        except Exception as e:
            traceback.print_exc()
            with _LOCK:
                job['status'] = 'error'
                job['error'] = '%s: %s' % (type(e).__name__, e)
        finally:
            with _LOCK:
                if _ACTIVE['id'] == job_id:
                    _ACTIVE['id'] = None
                job['finished'] = time.time()      # 冻结耗时，之后 elapsed 不再增长
                job['progress']['phase'] = 'finished'
            _persist_job(job)                      # 落盘存档：服务重启后导出仍可用
            _save_gpu_evidence(job, 'end')

    threading.Thread(target=_worker, daemon=True).start()
    return job_id


# =========================================================================== #
# 导出与取证
#   * 逐对明细导出报表（.xlsx，3 个 sheet）—— 零依赖手写 OOXML，
#     因为远端 conda 环境没有 openpyxl，手写可以少一个部署依赖。
#   * 匹配连线图按方法打包（424 张 PNG -> JPEG zip）
#   * GPU 运行证据（真实执行 nvidia-smi + torch 设备信息）
# 所有落盘都在 web/evidence/ 下，**不碰 result/**。
# =========================================================================== #
# 所有落盘都在 output_dir（.env 的 OUTPUT_DIR，宿主机可写挂载）下。
# 镜像里不给写权限也能启动（preflight 已先检查过），这里再兜一层清晰报错。
OUTPUT_DIR = Path(config_loader.output_dir(_CFG))
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    sys.stderr.write('[app] 输出目录不可用：%s（%s）\n' % (OUTPUT_DIR, e))
    sys.exit(1)
EVIDENCE_DIR = OUTPUT_DIR / 'evidence'
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_ARCHIVE_DIR = EVIDENCE_DIR / 'jobs'
JOBS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _default_job_id():
    """当前/最近一次评测的 job_id；没有就返回 ''（连线图退回离线 PNG 转码）。"""
    with _LOCK:
        jid = _ACTIVE.get('id')
    if jid:
        return jid
    arch = _latest_archive()
    return arch['id'] if arch else ''


def _matches_dir(job_id):
    """逐对 matches 存档目录：连线图的唯一可信数据源。

    只存这次评测跑出来的 matches（evidence/viz_matches/<job_id>/<method>.npz），
    不碰 result/。连线图必须拿它来画——事后重跑对 HLDD(XFeat) 会翻面，
    画出来的内点数会和逐对表格对不上。
    """
    return EVIDENCE_DIR / 'viz_matches' / job_id

VIZ_KINDS = ['matches', 'inliers', 'fusion', 'board']
VIZ_KIND_LABEL = {
    'matches': '全部匹配连线（绿 = 误差 ≤ 3px 判为正确；红 = 超过 3px）',
    'inliers': '仅内点连线（DEGENSAC 判为内点的匹配）',
    'fusion': '棋盘融合底图（看不出匹配关系，只作对照）',
    'board': '四宫格汇总图（matches / inliers / fusion / 叠加）',
}
VIZ_JPEG_QUALITY = 85
_VIZ_BUILDS = {}


def _human(n):
    if n is None:
        return None
    n = float(n)
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return ('%.0f %s' % (n, u)) if u == 'B' else ('%.1f %s' % (n, u))
        n /= 1024.0


# ---------------------------------- 工具 ---------------------------------- #
def _run_cmd(cmd, timeout=25):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return ((p.stdout or '') + (p.stderr or '')).rstrip()
    except Exception as e:
        return '（执行失败：%s: %s）' % (type(e).__name__, e)


def _png_to_jpeg_bytes(path=None, quality=VIZ_JPEG_QUALITY, src_bytes=None):
    """PNG -> JPEG 字节。RGBA 先按白底合成，避免透明区变黑。

    src_bytes 给出时从内存里的 PNG 字节转换（存档渲染的图不落盘，直接打包）。
    """
    from PIL import Image
    src = io.BytesIO(src_bytes) if src_bytes is not None else path
    with Image.open(src) as im:
        if im.mode in ('RGBA', 'LA', 'P'):
            im = im.convert('RGBA')
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert('RGB')
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=quality, optimize=True)
        return buf.getvalue()


# ------------------------------- 连线图打包 ------------------------------- #
def _viz_src_dir(method, kps, kind):
    """离线权威连线图目录（result/ 不进镜像、不进 git）。

    容器里通常没有它 —— 这不是错误：/api/viz 会走「评测存档」现渲染，
    zip 打包这条路只是**可选加速**（有现成 PNG 就直接转码）。
    想用就把它挂到 EM_OFFLINE_VIZ_DIR 上。
    """
    root = os.environ.get('EM_OFFLINE_VIZ_DIR') or (APP_ROOT / 'result')
    return Path(root) / method / str(kps) / 'VIS_SAR' / 'viz' / kind


def _viz_zip_path(method, kps, kind, fmt, job=None):
    # job 参与文件名：存档版和离线版的图内容不同，不能共用同一个 zip
    return EVIDENCE_DIR / 'viz_zips' / (
        '%s_%s_%s_%s_%s.zip' % (method, kps, kind, fmt, job or 'offline'))


def _archive_names(job, method):
    """该 job 存下来的样本名（排序）；没有存档/读不动返回 None。

    只按存档**实际有**的样本打包——存档是子集（比如只跑了 3 对）时，
    不能拿全量 424 个名字去渲染，否则缺的那些会静默降级成"重跑"，
    包就又变成重跑版了。
    """
    if not job:
        return None
    p = os.path.join(str(_matches_dir(job)), '%s.npz' % method)
    if not os.path.exists(p):
        return None
    try:
        import numpy as np
        with np.load(p) as z:
            return sorted(z.files)
    except Exception:
        return None


def _viz_zip_worker(key, method, kps, kind, fmt, job=None):
    st = _VIZ_BUILDS[key]
    try:
        # 存档模式：用这次评测存下来的 matches 画，与逐对表格同源
        arch_names = _archive_names(job, method) if kind == 'matches' else None
        use_archive = bool(arch_names)
        dst = Path(st['path'])
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + '.part')
        if part.exists():
            part.unlink()
        arc_dir = '%s_%s_%s' % (method, kps, kind)
        with zipfile.ZipFile(str(part), 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            if use_archive:
                md = str(_matches_dir(job))
                names = arch_names
                with _LOCK:
                    st['total'] = len(names)
                for i, nm in enumerate(names):
                    png = eval_core.render_matches(method, 'VIS_SAR', nm, kps,
                                                   matches_dir=md, job_label=job)
                    if fmt == 'jpg':
                        z.writestr('%s/%s.jpg' % (arc_dir, Path(nm).stem),
                                   _png_to_jpeg_bytes(src_bytes=png))
                    else:
                        z.writestr('%s/%s' % (arc_dir, Path(nm).stem + '.png'), png)
                    with _LOCK:
                        st['done'] = i + 1
            else:
                src = _viz_src_dir(method, kps, kind)
                pngs = sorted(glob.glob(str(src / '*.png')))
                if not pngs:
                    raise FileNotFoundError('连线图目录为空或不存在：%s' % src)
                with _LOCK:
                    st['total'] = len(pngs)
                for i, p in enumerate(pngs):
                    if fmt == 'jpg':
                        z.writestr('%s/%s.jpg' % (arc_dir, Path(p).stem),
                                   _png_to_jpeg_bytes(p))
                    else:
                        with open(p, 'rb') as fh:
                            z.writestr('%s/%s' % (arc_dir, Path(p).name), fh.read())
                    with _LOCK:
                        st['done'] = i + 1
        if dst.exists():
            dst.unlink()
        part.rename(dst)
        with _LOCK:
            st.update({'status': 'done', 'finished': time.time(),
                       'size': dst.stat().st_size})
    except Exception as e:
        traceback.print_exc()
        with _LOCK:
            st.update({'status': 'error', 'error': '%s: %s' % (type(e).__name__, e),
                       'finished': time.time()})


def _viz_build_start(method, kps, kind, fmt, job=None):
    key = (method, kps, kind, fmt, job)
    with _LOCK:
        st = _VIZ_BUILDS.get(key)
        if st and st['status'] == 'running':
            return dict(st)
        st = {'status': 'running', 'done': 0, 'total': 0, 'error': None, 'size': None,
              'started': time.time(), 'finished': None,
              'path': str(_viz_zip_path(method, kps, kind, fmt, job))}
        _VIZ_BUILDS[key] = st
    threading.Thread(target=_viz_zip_worker,
                     args=(key, method, kps, kind, fmt), kwargs={'job': job},
                     daemon=True).start()
    with _LOCK:
        return dict(_VIZ_BUILDS[key])


def _viz_status(method, kps, kind, fmt, job=None):
    key = (method, kps, kind, fmt, job)
    path = _viz_zip_path(method, kps, kind, fmt, job)
    ready = path.exists() and path.stat().st_size > 0
    with _LOCK:
        st = dict(_VIZ_BUILDS[key]) if key in _VIZ_BUILDS else None
    if st:
        st['elapsed'] = round((st.get('finished') or time.time()) - st['started'], 1)
    # 只有该 job 真有这个方法的 matches 存档才算 archive，否则退回离线 PNG 转码
    arch_names = _archive_names(job, method) if kind == 'matches' else None
    has_archive = bool(arch_names)
    return {
        'method': method, 'kps': kps, 'kind': kind, 'fmt': fmt,
        'label': VIZ_KIND_LABEL.get(kind, kind),
        'n_images': (len(arch_names) if has_archive
                     else len(glob.glob(str(_viz_src_dir(method, kps, kind) / '*.png')))),
        'source': ('archive' if has_archive else 'offline'),
        'ready': bool(ready),
        'size': (path.stat().st_size if ready else None),
        'size_h': (_human(path.stat().st_size) if ready else None),
        'mtime': (time.strftime('%Y-%m-%d %H:%M', time.localtime(path.stat().st_mtime))
                  if ready else None),
        'url': ('/api/export/viz?method=%s&kps=%d&kind=%s&fmt=%s%s'
                % (method, kps, kind, fmt, ('&job=%s' % job) if job else '')),
        'build': st,
    }


# ------------------------------ 手写 XLSX ------------------------------ #
_STYLES_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    '<fonts count="2">'
    '<font><sz val="11"/><color theme="1"/><name val="宋体"/></font>'
    '<font><b/><sz val="11"/><color theme="1"/><name val="宋体"/></font>'
    '</fonts>'
    '<fills count="2"><fill><patternFill patternType="none"/></fill>'
    '<fill><patternFill patternType="gray125"/></fill></fills>'
    '<borders count="1"><border/></borders>'
    '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
    '<cellXfs count="2">'
    '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
    '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
    '</cellXfs>'
    '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
    '</styleSheet>'
)


def _xml_escape(s):
    out = []
    for ch in str(s):
        o = ord(ch)
        if ch == '&':
            out.append('&amp;')
        elif ch == '<':
            out.append('&lt;')
        elif ch == '>':
            out.append('&gt;')
        elif ch == '"':
            out.append('&quot;')
        elif o < 0x20 and ch not in '\t\n\r':
            out.append(' ')
        else:
            out.append(ch)
    return ''.join(out)


def _xlsx_col(i):
    s = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _num_text(v):
    if v != v or v in (float('inf'), float('-inf')):   # nan / inf
        return None
    return ('%d' % v) if float(v).is_integer() else ('%.6f' % v)


def _xlsx_sheet(rows, widths=None):
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">']
    if widths:
        out.append('<cols>')
        for i, w in enumerate(widths, 1):
            out.append('<col min="%d" max="%d" width="%s" customWidth="1"/>' % (i, i, w))
        out.append('</cols>')
    out.append('<sheetData>')
    for ri, row in enumerate(rows, 1):
        cells = []
        for ci, v in enumerate(row):
            if hasattr(v, 'item') and not isinstance(v, (str, bytes)):
                try:
                    v = v.item()
                except Exception:
                    pass
            ref = '%s%d' % (_xlsx_col(ci), ri)
            st = ' s="1"' if ri == 1 else ''
            # 空值也要占位（写一个空 <c>），否则按位置解析的消费方会错列；
            # 行尾的占位最后统一去掉。
            if v is None or v == '':
                cells.append('<c r="%s"/>' % ref)
                continue
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, (int, float)):
                txt = _num_text(float(v))
                cells.append('<c r="%s"%s><v>%s</v></c>' % (ref, st, txt)
                             if txt is not None else '<c r="%s"/>' % ref)
            else:
                cells.append('<c r="%s"%s t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>'
                             % (ref, st, _xml_escape(v)))
        while cells and cells[-1].endswith('/>'):
            cells.pop()
        if cells:
            out.append('<row r="%d">%s</row>' % (ri, ''.join(cells)))
    out += ['</sheetData>', '</worksheet>']
    return ''.join(out)


def _xlsx_bytes(sheets):
    """sheets: [(名字, 二维数组, 列宽数组或 None)] -> .xlsx 字节流"""
    n = len(sheets)
    P = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
          '<Default Extension="xml" ContentType="application/xml"/>',
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
          '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    for i in range(1, n + 1):
        ct.append('<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % i)
    ct.append('</Types>')

    wb = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
          ' xmlns:r="%s"><sheets>' % P]
    for i, (name, _r, _w) in enumerate(sheets, 1):
        wb.append('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (_xml_escape(name), i, i))
    wb.append('</sheets></workbook>')

    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i in range(1, n + 1):
        rels.append('<Relationship Id="rId%d" Type="%s/worksheet" Target="worksheets/sheet%d.xml"/>' % (i, P, i))
    rels.append('<Relationship Id="rId%d" Type="%s/styles" Target="styles.xml"/>' % (n + 1, P))
    rels.append('</Relationships>')

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ''.join(ct))
        z.writestr('_rels/.rels',
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="%s/officeDocument" Target="xl/workbook.xml"/>'
                   '</Relationships>' % P)
        z.writestr('xl/workbook.xml', ''.join(wb))
        z.writestr('xl/_rels/workbook.xml.rels', ''.join(rels))
        z.writestr('xl/styles.xml', _STYLES_XML)
        for i, (_name, rows, widths) in enumerate(sheets, 1):
            z.writestr('xl/worksheets/sheet%d.xml' % i, _xlsx_sheet(rows, widths))
    return buf.getvalue()


# ---------------------------- 任务存档 / 取数 ---------------------------- #
def _persist_job(job):
    """把任务结果落盘，服务重启后导出仍可用（写在 evidence/jobs/，不碰 result/）。"""
    try:
        with _LOCK:
            payload = {
                'id': job['id'], 'status': job['status'], 'error': job['error'],
                'created': job['created'], 'finished': job.get('finished'),
                'subset': job['subset'], 'num_kps': job['num_kps'],
                'methods': job['methods'], 'n_pairs': job['n_pairs'],
                'total_pairs_in_subset': job['total_pairs_in_subset'],
                'scope_desc': job['scope_desc'], 'scope': job.get('scope'),
                'summaries': {m: (job['results'][m]['summary'] if job['results'].get(m) else None)
                              for m in job['methods']},
                'pairs': {m: list(job['results'][m]['pairs'])
                          for m in job['methods'] if job['results'].get(m)},
            }
        p = JOBS_ARCHIVE_DIR / ('%s.json' % job['id'])
        with open(str(p), 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False)
        return p
    except Exception as e:
        traceback.print_exc()
        print('[web] 任务存档失败：%s: %s' % (type(e).__name__, e), flush=True)
        return None


def _load_job(job_id):
    """先查内存，再回退到 evidence/jobs/ 的存档。"""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            return {
                'id': job['id'], 'status': job['status'], 'error': job['error'],
                'created': job['created'], 'finished': job.get('finished'),
                'subset': job['subset'], 'num_kps': job['num_kps'],
                'methods': job['methods'], 'n_pairs': job['n_pairs'],
                'total_pairs_in_subset': job['total_pairs_in_subset'],
                'scope_desc': job['scope_desc'], 'scope': job.get('scope'),
                'summaries': {m: (job['results'][m]['summary'] if job['results'].get(m) else None)
                              for m in job['methods']},
                'pairs': {m: list(job['results'][m]['pairs'])
                          for m in job['methods'] if job['results'].get(m)},
                'from_archive': False,
            }
    p = JOBS_ARCHIVE_DIR / ('%s.json' % job_id)
    if p.exists():
        try:
            with open(str(p), 'r', encoding='utf-8') as fh:
                d = json.load(fh)
            d['from_archive'] = True
            return d
        except Exception:
            traceback.print_exc()
    return None


# ------------------------------ GPU 运行证据 ------------------------------ #
def _gpu_evidence(job=None):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    L = []
    L.append('VIS_SAR 数据集评测 · GPU 运行证据')
    L.append('=' * 78)
    L.append('生成时间              : %s' % ts)
    L.append('主机                  : %s' % socket.gethostname())
    L.append('服务进程 PID          : %d' % os.getpid())
    L.append('服务命令行            : %s' % ' '.join(sys.argv))
    L.append('工作目录              : %s' % os.getcwd())
    L.append('监听端口              : %s' % ((_CFG.get('server') or {}).get('port')
                                              or os.environ.get('PORT') or 8000))
    L.append('CUDA_VISIBLE_DEVICES  : %s'
             % os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置 = 全部可见)'))
    if job is not None:
        L.append('')
        L.append('本次评测任务')
        L.append('-' * 78)
        L.append('job_id                : %s' % job['id'])
        L.append('评测范围              : %s' % job['scope_desc'])
        L.append('关键点数              : %s' % job['num_kps'])
        L.append('被测方法（%d 个）      : %s'
                 % (len(job['methods']), ', '.join(disp_name(m) for m in job['methods'])))
        L.append('图像对数              : %s' % job['n_pairs'])
    L.append('')
    L.append('torch / CUDA')
    L.append('-' * 78)
    try:
        import torch
        L.append('torch 版本            : %s' % torch.__version__)
        L.append('CUDA 可用             : %s' % torch.cuda.is_available())
        if torch.cuda.is_available():
            L.append('对进程可见的设备数    : %d' % torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                L.append('  逻辑设备 %d         : %s (sm_%d%d)'
                         % (i, torch.cuda.get_device_name(i), cap[0], cap[1]))
            L.append('当前设备              : cuda:%d' % torch.cuda.current_device())
            free, total = torch.cuda.mem_get_info()
            L.append('整卡显存 空闲 / 总量  : %.0f / %.0f MiB'
                     % (free / 1048576.0, total / 1048576.0))
            L.append('本进程显存 已分配     : %.1f MiB' % (torch.cuda.memory_allocated() / 1048576.0))
            L.append('本进程显存 已保留     : %.1f MiB' % (torch.cuda.memory_reserved() / 1048576.0))
            L.append('cuDNN 版本            : %s' % torch.backends.cudnn.version())
    except Exception as e:
        L.append('（读取 torch 信息失败：%s: %s）' % (type(e).__name__, e))
    L.append('')
    L.append('nvidia-smi --query-gpu（本机全部 GPU）')
    L.append('-' * 78)
    L.append(_run_cmd(['nvidia-smi',
                       '--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu',
                       '--format=csv']))
    L.append('')
    L.append('nvidia-smi --query-compute-apps（占用 GPU 的进程）')
    L.append('-' * 78)
    L.append(_run_cmd(['nvidia-smi',
                       '--query-compute-apps=pid,process_name,used_memory',
                       '--format=csv']))
    L.append('')
    L.append('nvidia-smi（完整输出）')
    L.append('-' * 78)
    L.append(_run_cmd(['nvidia-smi']))
    L.append('')
    L.append('=' * 78)
    L.append('说明：以上为服务进程内真实执行的命令输出，未经加工。')
    return '\n'.join(L)


def _save_gpu_evidence(job=None, tag='snapshot'):
    """把 GPU 证据落盘到 evidence/，供测评报告取证（不碰 result/）。"""
    try:
        text = _gpu_evidence(job)
        name = 'gpu_%s_%s.txt' % (job['id'] if job else 'current', tag)
        p = EVIDENCE_DIR / name
        with open(str(p), 'w', encoding='utf-8') as fh:
            fh.write(text)
        print('[web] GPU 证据已落盘：%s' % p, flush=True)
        return p
    except Exception as e:
        traceback.print_exc()
        print('[web] GPU 证据落盘失败：%s: %s' % (type(e).__name__, e), flush=True)
        return None


def _env_rows(job):
    rows = [['项目', '值']]
    rows += [
        ['生成时间', time.strftime('%Y-%m-%d %H:%M:%S')],
        ['任务 ID', job['id']],
        ['评测范围', job['scope_desc']],
        ['关键点数', job['num_kps']],
        ['被测方法数', len(job['methods'])],
        ['被测方法', ', '.join(disp_name(m) for m in job['methods'])],
        ['数据集', '%s test 集' % job['subset']],
        ['本次评测图像对数', job['n_pairs']],
        ['正确匹配判据', '匹配点到真值单应投影偏差 ≤ %s px' % eval_core.SUCCESS_THRES],
        ['成功帧判据', '帧内正确匹配数 ≥ %s' % eval_core.MIN_NCM],
        ['单应估计方法', 'pydegensac.findHomography，内点阈值 2 px'],
        ['CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')],
    ]
    try:
        import torch
        rows.append(['torch 版本', torch.__version__])
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            rows.append(['GPU 型号', torch.cuda.get_device_name(0)])
            rows.append(['GPU 算力', 'sm_%d%d' % (cap[0], cap[1])])
    except Exception as e:
        rows.append(['torch 信息', '读取失败：%s' % e])
    rows += [
        ['', ''],
        ['可复现性说明',
         'Success Rate / Average Matches / NCM / ME / RMSE / MMA 只用真值单应计算，是确定性的、可逐位复现；'
         'Homography 依赖随机采样算法 DEGENSAC，每次运行会有少量图在阈值附近翻面，'
         '差值恒为 1/样本数 的整数倍，不应作为精确值引用。'],
        ['统计范围说明',
         'NCM / ME / RMSE 只在成功帧（帧内正确匹配数 ≥ %s）上统计；'
         'ME 与 RMSE 为「先逐帧计算、再对成功帧取平均」的两层平均；'
         'Average Matches 把匹配失败的图像对按 0 计入分母；'
         'Average Time 丢弃第一条计时记录（规避设备预热）。' % eval_core.MIN_NCM],
    ]
    return rows


def _job_to_xlsx(job):
    """把一次任务导出成 3 个 sheet 的 xlsx 字节流。"""
    # 「真值方向」（suffix，如 .21/.12）按要求不进报表：
    # 它只是样本文件名的后缀，对判读结果没有帮助，导出时去掉这一列。
    det = [['方法', '图像对', '匹配数', '正确匹配 ncm', '是否成功帧',
            'ME (px)', 'RMSE (px)', '单应角点距离 (px)', '匹配耗时 (s)', '备注']]
    for m in job['methods']:
        for r in job.get('pairs', {}).get(m, []):
            det.append([
                disp_name(m), r.get('name'), r.get('n_matches'), r.get('ncm'),
                '是' if r.get('success') else '否',
                r.get('me'), r.get('rmse'), r.get('corner_dist'), r.get('match_time'),
                (r.get('error') or ''),
            ])

    smy = [['方法', 'Success Rate', 'Average Matches', 'NCM', 'ME (px)', 'RMSE (px)',
            'Average Time (s)', '成功帧数', '失败数']]
    for m in job['methods']:
        s = job.get('summaries', {}).get(m) or {}
        pairs = job.get('pairs', {}).get(m, [])
        n_ok = sum(1 for r in pairs if r.get('success'))
        smy.append([disp_name(m), s.get('SuccessRate'), s.get('AverageMatches'), s.get('NCM'),
                    s.get('ME'), s.get('RMSE'), s.get('AverageTime'), n_ok,
                    s.get('failed_count', len(pairs) - n_ok)])

    return _xlsx_bytes([
        ('逐对明细', det, [10, 12, 10, 15, 12, 11, 12, 19, 14, 40]),
        ('方法汇总', smy, [10, 14, 17, 12, 11, 12, 17, 11, 10]),
        ('运行环境', _env_rows(job), [24, 110]),
    ])


# --------------------------------------------------------------------------- #
# 页面本身由 client 容器（nginx）提供，后端只认 /api。
# 保留 / 是为了「直接访问后端端口」时不至于看到 404 以为服务挂了。
@app.route('/')
def index():
    return jsonify({
        'service': 'evalmatcher-server',
        'page': '前端在同机的 nginx（.env 的 WEB_PORT）上，或按 README 单独部署 client',
        'health': '/api/health',
    })


@app.route('/api/health')
def api_health():
    """健康检查：只读地看几个关键项，不加载模型（所以要快）。"""
    import torch
    dev = (_CFG.get('server') or {}).get('device', 'cuda')
    try:
        cuda = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
    except Exception as e:
        cuda, gpu_name = False, '%s: %s' % (type(e).__name__, e)
    ds = Path(config_loader.dataset_dir(_CFG))
    return jsonify({
        'status': 'ok',
        'config': _CFG['_config_path'],
        'methods': METHODS_ALL,
        'labels': LABELS,
        'subsets': SUBSETS,
        'device': dev,
        'cuda_available': cuda,
        'gpu_name': gpu_name,
        'dataset_dir': str(ds),
        'dataset_readable': ds.is_dir(),
        'output_dir': str(OUTPUT_DIR),
        'output_writable': os.access(str(OUTPUT_DIR), os.W_OK),
    })


@app.route('/api/config')
def api_config():
    return jsonify({
        'methods': METHODS_ALL, 'labels': LABELS, 'name2id': NAME2ID, 'subsets': SUBSETS,
        'num_kps_choices': NUM_KPS_CHOICES,
        'success_thres': eval_core.SUCCESS_THRES, 'min_ncm': eval_core.MIN_NCM,
        'total_pairs': {s: len(eval_core.image_names(s)) for s in SUBSETS},
    })


@app.route('/api/pairs')
def api_pairs():
    subset = request.args.get('subset', 'VIS_SAR')
    names = eval_core.image_names(subset)
    return jsonify({'subset': subset, 'names': names, 'total': len(names)})


@app.route('/api/image')
def api_image():
    subset = request.args.get('subset', 'VIS_SAR')
    side = request.args.get('side', 'vis')
    name = request.args.get('name', '')
    if subset not in SUBSETS or name not in set(eval_core.image_names(subset)):
        return jsonify({'error': '样本不存在'}), 404
    folder = subset.split('_')[0] if side == 'vis' else subset.split('_')[1]
    return send_from_directory(
        str(Path(config_loader.dataset_dir(_CFG)) / subset / 'test' / folder), name)


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    if _ACTIVE['id']:
        return jsonify({'error': '已有评测在跑，请等它结束',
                        'job_id': _ACTIVE['id']}), 409
    payload = request.get_json(silent=True) or {}
    try:
        job_id = _start_job(payload)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    with _LOCK:
        job = _JOBS[job_id]
        return jsonify({'job_id': job_id, 'seed': {
            'subset': job['subset'], 'methods': job['methods'],
            'n_pairs': job['n_pairs'], 'scope_desc': job['scope_desc'],
            'num_kps': job['num_kps']}})


def _latest_archive():
    """evidence/jobs/ 里最近一份存档（按 created 取最大，别按文件名——job_id 是随机的）。"""
    out = []
    for p in glob.glob(str(JOBS_ARCHIVE_DIR / '*.json')):
        try:
            with open(p, 'r', encoding='utf-8') as fh:
                d = json.load(fh)
        except Exception:
            continue
        if d.get('id') and d.get('created'):
            out.append(d)
    return max(out, key=lambda d: d['created']) if out else None


@app.route('/api/last_job')
def api_last_job():
    """最近一次评测的 job_id —— 供页面刷新/换浏览器后接回结果用。

    页面优先读自己的 localStorage；读不到（清过缓存、换了设备）时退到这里。
    只返回 id，不返回结果——结果仍走 /api/job，避免两套返回结构。

    服务重启后内存 _JOBS 是空的，这里退回 evidence/jobs/ 的最新存档，
    否则"接回这次结果"的入口会跟着重启一起消失。
    """
    with _LOCK:
        if _JOBS:
            newest = max(_JOBS.values(), key=lambda j: j['created'])
            return jsonify({
                'job_id': newest['id'], 'status': newest['status'],
                'scope_desc': newest['scope_desc'], 'num_kps': newest['num_kps'],
                'methods': newest['methods'], 'n_pairs': newest['n_pairs'],
                'scope': newest.get('scope'), 'created': newest['created'],
                'from_archive': False,
            })
    arch = _latest_archive()
    if arch is None:
        return jsonify({'job_id': None, 'reason': 'no_jobs'})
    return jsonify({
        'job_id': arch.get('id'), 'status': arch.get('status'),
        'scope_desc': arch.get('scope_desc'), 'num_kps': arch.get('num_kps'),
        'methods': arch.get('methods'), 'n_pairs': arch.get('n_pairs'),
        'scope': arch.get('scope'), 'created': arch.get('created'),
        'from_archive': True,
    })


def _archive_snapshot(arch, pairs_for=None):
    """服务重启后，把 evidence/jobs/ 里的存档还原成 /api/job 的返回结构。

    为什么需要：_JOBS 只在内存里，一重启就空了。若 /api/job 只查内存，
    页面「接回这次结果」会拿到 404 -> doRestore 把 localStorage 里的 jobId 清掉，
    用户上一次（往往要跑 6 分钟的）结果从此再也看不回来。
    """
    methods = arch.get('methods') or []
    n_pairs = arch.get('n_pairs') or 0
    snap = {
        'id': arch.get('id'), 'status': arch.get('status'), 'error': arch.get('error'),
        'subset': arch.get('subset'), 'num_kps': arch.get('num_kps'),
        'methods': methods, 'n_pairs': n_pairs,
        'total_pairs_in_subset': arch.get('total_pairs_in_subset'),
        'scope_desc': arch.get('scope_desc'), 'scope': arch.get('scope'),
        'elapsed': round((arch.get('finished') or arch.get('created') or 0)
                         - (arch.get('created') or 0), 1),
        'progress': {'method': None, 'method_index': -1, 'n_methods': len(methods),
                     'pair_index': -1, 'pair_name': None,
                     'pairs_done': n_pairs * len(methods),
                     'pairs_total': n_pairs * len(methods), 'phase': 'finished'},
        'summaries': arch.get('summaries') or {},
        'n_done_pairs': {m: len(v) for m, v in (arch.get('pairs') or {}).items()},
        'events': ['（服务重启过，结果由 evidence/jobs 存档恢复，实时日志不再保留）'],
        'from_archive': True,
    }
    if pairs_for and (arch.get('pairs') or {}).get(pairs_for):
        snap['pairs'] = list(arch['pairs'][pairs_for])
        snap['pairs_for'] = pairs_for
    return snap


@app.route('/api/job')
def api_job():
    job_id = request.args.get('id', '')
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is not None:
        return jsonify(_snapshot(job, pairs_for=request.args.get('pairs_for')))
    arch = _load_job(job_id)          # 内存没有 -> 回退 evidence/jobs/ 存档
    if arch is None:
        return jsonify({'error': '任务不存在或已过期'}), 404
    return jsonify(_archive_snapshot(arch, pairs_for=request.args.get('pairs_for')))


@app.route('/api/viz')
def api_viz():
    subset = request.args.get('subset', 'VIS_SAR')
    name = request.args.get('name', '')
    method = request.args.get('method', '')
    num_kps = int(request.args.get('num_kps', 1024))
    if name not in set(eval_core.image_names(subset)) or method not in METHODS_ALL:
        return jsonify({'error': '参数不合法'}), 400
    # 带 job 时优先用该次评测存下来的 matches 画（与逐对表格同源）；
    # 没带 job 或存档不存在才现场重跑，并在响应头 X-Viz-Source 里标明。
    job_id = request.args.get('job', '')
    md = str(_matches_dir(job_id)) if job_id else None
    info = {}
    try:
        png = eval_core.render_matches(method, subset, name, num_kps,
                                       matches_dir=md, job_label=job_id or '',
                                       info=info)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': '%s: %s' % (type(e).__name__, e)}), 500
    resp = Response(png, mimetype='image/png')
    resp.headers['X-Viz-Source'] = info.get('source', 'rerun')
    resp.headers['X-Viz-NCM'] = str(info.get('ncm'))
    resp.headers['Cache-Control'] = 'no-store'   # 存档/重跑都会变，别让浏览器缓存住
    return resp


# =========================== 导出：逐对明细报表 =========================== #
@app.route('/api/export/xlsx')
def api_export_xlsx():
    """逐对明细导出报表（3 个 sheet：逐对明细 / 方法汇总 / 运行环境）。"""
    job_id = request.args.get('job', '')
    job = _load_job(job_id) if job_id else None
    if job is None:
        return jsonify({'error': '任务不存在（服务重启后可从 /api/evidence/jobs 里找存档）'}), 404
    try:
        blob = _job_to_xlsx(job)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': '%s: %s' % (type(e).__name__, e)}), 500
    fname = 'VIS_SAR_逐对明细_%s_%skps_%s.xlsx' % (
        job['id'], job['num_kps'], time.strftime('%Y%m%d-%H%M'))
    return Response(
        blob,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': "attachment; filename*=UTF-8''%s"
                 % quote(fname)})


@app.route('/api/evidence/jobs')
def api_evidence_jobs():
    """已落盘的任务存档（服务重启后导出仍可用）。"""
    out = []
    for p in sorted(glob.glob(str(JOBS_ARCHIVE_DIR / '*.json')), reverse=True):
        try:
            with open(p, 'r', encoding='utf-8') as fh:
                d = json.load(fh)
            s = {}
            for m in d.get('methods', []):
                sm = (d.get('summaries') or {}).get(m) or {}
                s[m] = {'SuccessRate': sm.get('SuccessRate'), 'NCM': sm.get('NCM')}
            out.append({'job_id': d.get('id'), 'created': d.get('created'),
                        'finished': d.get('finished'), 'status': d.get('status'),
                        'scope_desc': d.get('scope_desc'), 'num_kps': d.get('num_kps'),
                        'methods': d.get('methods'),
                        'n_pairs_per_method': {m: len(v)
                                               for m, v in (d.get('pairs') or {}).items()},
                        'summary': s,
                        'size': os.path.getsize(p),
                        'size_h': _human(os.path.getsize(p))})
        except Exception:
            continue
    return jsonify({'jobs': out})


# ========================= 导出：匹配连线图打包 ========================= #
@app.route('/api/exports')
def api_exports():
    kind = request.args.get('kind', 'matches')
    fmt = request.args.get('fmt', 'jpg')
    kps = int(request.args.get('kps', 1024))
    if kind not in VIZ_KINDS:
        return jsonify({'error': '不支持的图种：%s' % kind}), 400
    with _LOCK:
        job_id = max(_JOBS.values(), key=lambda j: j['created'])['id'] if _JOBS else None
    return jsonify({
        'kps': kps, 'kind': kind, 'fmt': fmt,
        'kind_label': VIZ_KIND_LABEL.get(kind),
        'kinds': VIZ_KINDS,
        'latest_job': job_id,
        'viz': {m: _viz_status(m, kps, kind, fmt, _default_job_id())
                for m in METHODS_ALL},
    })


@app.route('/api/export/viz')
def api_export_viz():
    """某方法的 424 张匹配连线图打包下载。

    首次请求若 zip 还没生成 -> 起后台线程转换（424 张约几十秒），
    立刻返回 202 + 进度，前端轮询 /api/exports 到 ready 再点下载。
    """
    method = request.args.get('method', '')
    kind = request.args.get('kind', 'matches')
    fmt = request.args.get('fmt', 'jpg')
    kps = int(request.args.get('kps', 1024))
    # 带 job = 用这次评测存下来的 matches 画（与逐对表格同源）；不带则退回离线 PNG 转码
    job = request.args.get('job', '') or _default_job_id()
    if method not in METHODS_ALL or kind not in VIZ_KINDS:
        return jsonify({'error': '参数不合法'}), 400
    if fmt not in ('jpg', 'png'):
        return jsonify({'error': 'fmt 只支持 jpg / png'}), 400

    path = _viz_zip_path(method, kps, kind, fmt, job)
    if path.exists() and path.stat().st_size > 0:
        return send_file(str(path), mimetype='application/zip', as_attachment=True,
                         download_name=path.name)
    st = _viz_build_start(method, kps, kind, fmt, job)
    return jsonify({'status': st['status'], 'method': method, 'kps': kps,
                    'kind': kind, 'fmt': fmt, 'job': job,
                    'note': '正在后台打包，请轮询 /api/exports 直到 ready'}), 202


# =========================== GPU 运行证据 =========================== #
@app.route('/api/evidence/gpu')
def api_evidence_gpu():
    """真实执行 nvidia-smi + 读取 torch 设备信息，返回纯文本（供截图取证）。"""
    job_id = request.args.get('job', '')
    job = _load_job(job_id) if job_id else None
    text = _gpu_evidence(job)
    if request.args.get('save') == '1':
        tag = (job_id or 'current') + '_' + time.strftime('%Y%m%d-%H%M%S')
        p = EVIDENCE_DIR / ('gpu_%s.txt' % tag)
        try:
            with open(str(p), 'w', encoding='utf-8') as fh:
                fh.write(text)
        except Exception:
            traceback.print_exc()
    if request.args.get('download') == '1':
        fname = 'GPU运行证据_%s_%s.txt' % (job_id or 'current',
                                          time.strftime('%Y%m%d-%H%M'))
        return Response(text, mimetype='text/plain; charset=utf-8',
                        headers={'Content-Disposition': "attachment; filename*=UTF-8''%s"
                                 % quote(fname)})
    return Response(text, mimetype='text/plain; charset=utf-8')


if __name__ == '__main__':
    # 端口以 config/server.yaml 为准（容器里固定 8000，对外端口由 compose 映射）。
    # PORT 只是开发机上 ./scripts/run_dev.sh 的便利开关。
    srv = _CFG.get('server') or {}
    port = int(os.environ.get('PORT') or srv.get('port') or 8000)
    host = srv.get('host') or '0.0.0.0'
    print('[server] app_root=%s config=%s port=%s cuda_visible=%s device=%s' %
          (APP_ROOT, _CFG['_config_path'], port,
           os.environ.get('CUDA_VISIBLE_DEVICES', '(all)'), srv.get('device')),
          flush=True)
    # 注意：任务表 _JOBS 是**进程内存**里的 dict，所以用 gunicorn 时必须
    # --workers 1，否则轮询 /api/job 可能打到另一个 worker 上找不到任务。
    app.run(host=host, port=port, threaded=True)
