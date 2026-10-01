import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from third.redfeat.lib.model import MMNet

import scipy.io as scio
from copy import deepcopy
import time

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# device-agnostic: 原本三处硬编码 .cuda()，在无 GPU 机器上直接崩。
# 有 CUDA 时仍是 cuda:0，行为不变；无 CUDA 时退回 CPU。
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AP = nn.AvgPool2d(9, stride=1, padding=4).to(DEVICE)
MP = nn.AvgPool2d(9, stride=1, padding=4).to(DEVICE)


def load_network(model_fn):
    checkpoint = torch.load(model_fn, map_location='cpu', weights_only=False)
    model = MMNet()
    weights = checkpoint['model']
    model.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return model.eval()


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rep_thr=0.6):
        super(NonMaxSuppression, self).__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rep_thr = rep_thr

    def forward(self, repeatability):
        # repeatability = repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        border_mask = maxima * 0
        border_mask[:, :, 10:-10, 10:-10] = 1
        maxima = maxima * border_mask
        # print(maxima.sum())
        return maxima.nonzero().t()[2:4]


def extract_multiscale(net, img, detector, image_type,
                       scale_f=2 ** 0.25, min_scale=0.0,
                       max_scale=1, min_size=256,
                       max_size=1024, verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"

    assert max_scale <= 1
    s = 1.0  # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []

    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")

            mask_extra = (MP((img > 1e-12).sum(dim=1, keepdim=True).float()) > 1e-5).float()
            for ii in range(3):
                mask_extra = (AP(mask_extra) > 0.99).float()

            img_t = (img - img.mean(dim=[-1, -2], keepdim=True)) / img.std(dim=[-1, -2], keepdim=True)

            with torch.no_grad():
                if image_type == '1':
                    descriptors, repeatability = net.forward1(img_t)
                elif image_type == '2':
                    descriptors, repeatability = net.forward2(img_t)

            mask = repeatability * 0
            mask[:, :, args.border:-args.border, args.border:-args.border] = 1
            repeatability = repeatability * mask * mask_extra
            y, x = detector(repeatability)  # nms
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]
            # accumulate multiple scales
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            # S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    # S = torch.cat(S) # scale
    scores = torch.cat(Q)  # scores = reliability * repeatability
    XYS = torch.stack([X, Y], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores



import argparse

parser = argparse.ArgumentParser("Extract keypoints for a given image")
parser.add_argument("--subsets", type=str, default='VIS_NIR', help='VIS_IR, VIS_NIR, VIS_SAR')
parser.add_argument("--num_features", type=int, default=4096, help='Number of features')
parser.add_argument("--model", type=str, default='ReDFeat_NIR/50.pth', help='model path')

parser.add_argument("--scale-f", type=float, default=2 ** 1)
parser.add_argument("--min-size", type=int, default=256)
parser.add_argument("--max-size", type=int, default=1000)
parser.add_argument("--min-scale", type=float, default=0)
parser.add_argument("--max-scale", type=float, default=1)
parser.add_argument("--border", type=float, default=5)
parser.add_argument("--repeatability-thr", type=float, default=0.01)

parser.add_argument("--gpu", type=int, default=0, help='use -1 for CPU')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
args = parser.parse_args()


def extract(imgpath1, imgpath2, num_kps, subset):
    args.num_features = num_kps
    if subset == 'VIS_NIR' and num_kps < 4096:
        args.scale_f = 2 ** 1.3
    model_path = 'third/redfeat/Pretrained/' + subset + '.pth'
    net = load_network(model_path)
    net = net.to(DEVICE)
    # create the non-maxima detector
    detector = NonMaxSuppression(
        rep_thr=args.repeatability_thr)

    img = Image.open(imgpath1).convert('RGB')
    W, H = img.size
    img = TF.to_tensor(img).unsqueeze(0)
    # img = (img-img.mean(dim=[-1,-2],keepdim=True))/img.std(dim=[-1,-2],keepdim=True)
    img = img.to(DEVICE)
    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector, '1',
                                           scale_f=args.scale_f,
                                           min_scale=args.min_scale,
                                           max_scale=args.max_scale,
                                           min_size=args.min_size,
                                           max_size=args.max_size,
                                           verbose=False)
    if len(scores) < args.num_features:
        idxs = scores.topk(len(scores))[1]
    else:
        idxs = scores.topk(args.num_features)[1]
    kp1 = xys[idxs]
    desc1 = desc[idxs]

    img = Image.open(imgpath2).convert('RGB')
    W, H = img.size
    img = TF.to_tensor(img).unsqueeze(0)
    # img = (img-img.mean(dim=[-1,-2],keepdim=True))/img.std(dim=[-1,-2],keepdim=True)
    img = img.to(DEVICE)

    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector, '2',
                                           scale_f=args.scale_f,
                                           min_scale=args.min_scale,
                                           max_scale=args.max_scale,
                                           min_size=args.min_size,
                                           max_size=args.max_size,
                                           verbose=False)
    if len(scores) < args.num_features:
        idxs = scores.topk(len(scores))[1]
    else:
        idxs = scores.topk(args.num_features)[1]
    kp2 = xys[idxs]
    desc2 = desc[idxs]

    return kp1, kp2, desc1, desc2



