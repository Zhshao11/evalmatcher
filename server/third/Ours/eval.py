import sys
import os
import time

import pydegensac

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from scipy.io import savemat
import numpy as np
import cv2
import os
import torch
from PIL import Image
from tqdm import tqdm
from skimage.feature import match_descriptors
import scipy.io as scio
import argparse
from third.Ours.lib.xfeat import XFeat
from third.Ours.lib.utils import cal_reproj_dists_H, visual_matching

method_name = 'Ours_2024_09_12-10_47_51_940000'
xfeat = XFeat(weights='trained_ckpt/model/2024_09_12-10_47_51/VIS_SAR_94000.pth')
blacklist_NIR = ['89.png', '87.png', '105.png', '129.png']
subset = 'VIS_SAR'
nums_kp = 4096
vis_flag = False
Matching = True
Homography = True

ransac_thres = 2
thres = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
match_failed = 0
n_matches = []
match_time = []
match_errs = []
thres_range = np.arange(1, 16)
thre_err = {thr: 0 for thr in thres_range}
scale = np.ones(4)

inlier_ratio = []
h_failed = 0
dists_sa = []

subset_path = os.path.join(SCRIPT_DIR, subset)
filepath1 = os.path.join(subset_path, 'test', subset.split('_')[0])
filepath2 = os.path.join(subset_path, 'test', subset.split('_')[1])
image_list = sorted(os.listdir(filepath1))
img_list_whitelist = []
progress_bar = tqdm(range(len(image_list)))

img_nums = 0
all_NCM = []
SR = 0
for id in progress_bar:
    img_nums += 1

    if subset == 'NIR' and image_list[id] in blacklist_NIR:
        continue
    else:
        img_list_whitelist.append(image_list[id])
    imgpath1 = os.path.join(filepath1, image_list[id])
    imgpath2 = os.path.join(filepath2, image_list[id])

    im1 = cv2.imread(imgpath1)
    im2 = cv2.imread(imgpath2)

    # Predict matches
    try:
        t0 = time.time()
        matches, kp1, kp2 = xfeat.match_cmodel(im1, im2, top_k=4096)
        match_time.append(time.time() - t0)
        matches = matches[0].cpu().numpy()
    except Exception as e:
        print(e)
        p1s = p2s = matches = []
        match_failed += 1
    n_matches.append(len(matches))

    try:
        suffix = '.12'
        H_gt = scio.loadmat(os.path.join(subset_path, 'test', 'transforms') + '/' + image_list[id].replace('.png', suffix + '.mat'))['H']

        if Matching:
            if len(matches) == 0:
                dist = np.array([float("inf")])
            else:
                dist = cal_reproj_dists_H(matches[:, 2:], matches[:, :2], H_gt)
            match_errs.append(dist)
            for thr in thres_range:
                thre_err[thr] += np.mean(dist <= thr)

        if Homography:
            try:
                H_pred, inliers = pydegensac.findHomography(matches[:, 2:], matches[:, :2], ransac_thres)
            except:
                H_pred = None

            if H_pred is None:
                corner_dist = np.nan
                h_failed += 1
                irat = 0
                inliers = []
            else:
                im = Image.open(imgpath1)
                w, h = im.size
                w, h = w / scale[0], h / scale[1]
                corners = np.array([[0, 0, 1],
                                    [0, h - 1, 1],
                                    [w - 1, 0, 1],
                                    [w - 1, h - 1, 1]])
                real_warped_corners = np.dot(corners, np.transpose(H_gt))
                real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
                irat = np.mean(inliers)

            if sum(inliers) >= 4:
                SR += 1
            all_NCM.append(sum(inliers))
            inlier_ratio.append(irat)
            dists_sa.append(corner_dist)
    except:
        suffix = '.21'
        H_gt = scio.loadmat(os.path.join(subset_path, 'test', 'transforms') + '/' + image_list[id].replace('.png', suffix + '.mat'))['H']

        if Matching:
            if len(matches) == 0:
                dist = np.array([float("inf")])
            else:
                dist = cal_reproj_dists_H(matches[:, :2], matches[:, 2:], H_gt)
            match_errs.append(dist)
            for thr in thres_range:
                thre_err[thr] += np.mean(dist <= thr)

        if Homography:
            try:
                H_pred, inliers = pydegensac.findHomography(matches[:, :2], matches[:, 2:], ransac_thres)
            except:
                H_pred = None

            if H_pred is None:
                corner_dist = np.nan
                h_failed += 1
                irat = 0
                inliers = []
            else:
                im = Image.open(imgpath1)
                w, h = im.size
                w, h = w / scale[0], h / scale[1]
                corners = np.array([[0, 0, 1],
                                    [0, h - 1, 1],
                                    [w - 1, 0, 1],
                                    [w - 1, h - 1, 1]])
                real_warped_corners = np.dot(corners, np.transpose(H_gt))
                real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
                irat = np.mean(inliers)

            if sum(inliers) >= 4:
                SR += 1
            all_NCM.append(sum(inliers))
            inlier_ratio.append(irat)
            dists_sa.append(corner_dist)

    if vis_flag:
        if len(inliers) > 0:
            visual_matching(method_name, subset, image_list[id], im1, im2, matches[inliers], H_pred, suffix)
    # if img_nums > 2:
    #     break

out_root = 'result/' + method_name + '/'
os.makedirs(out_root, exist_ok=True)
f = open((out_root + subset + '.txt'), 'a+')

# print MMA
thres = np.array(thres)
ierr = np.array([thre_err[th] / (img_nums) for th in thres])
f.write('#### MMA ####')
f.write('\n')
for i in range(len(ierr)):
    f.write(str(format(ierr[i], ".5f")))
    f.write(' ')

f.write('\n')
f.write('\n')

# print Homography
correct_sa = np.mean([[float(d <= t) for t in thres] for d in dists_sa], axis=0)
f.write('#### Homography ####')
f.write('\n')
for i in range(len(correct_sa)):
    f.write(str(format(correct_sa[i], ".5f")))
    f.write(' ')

f.write('\n')
f.write('\n')

# print Success Rate
SR = SR / img_nums
f.write('#### Success Rate ####')
f.write('\n')
f.write(str(format(SR, ".5f")))

f.write('\n')
f.write('\n')

# print Average_Matches
AM = np.mean(n_matches)
f.write('#### Average Matches ####')
f.write('\n')
f.write(str(format(AM, ".5f")))

f.write('\n')
f.write('\n')

# print NCM
NCM = np.mean(all_NCM)
f.write('#### NCM ####')
f.write('\n')
f.write(str(format(NCM, ".5f")))

f.write('\n')
f.write('\n')

# print Average_Time
AT = np.mean(match_time[1:])
f.write('#### Average Time ####')
f.write('\n')
f.write(str(format(AT, ".5f")))

f.close()



