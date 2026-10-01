import sys
import os
import time

import pydegensac
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from scipy.io import savemat
import numpy as np
import cv2
import os
import torch
from PIL import Image
from tqdm import tqdm
import scipy.io as scio
from utils import cal_reproj_dists_H, visual_matching
import matcher

method_names = ['d2net', 'redfeat', 'Ours', 'XoFTR', 'LoFTR']
subsets = ['VIS_SAR']
Nums_kps = [1024]
vis_flag = True
Matching = True
Homography = True

blacklist_NIR = ['89.png', '105.png', '115.png', '129.png']
for method_name in method_names:
    for subset in subsets:
        for nums_kps in Nums_kps:
            if method_name == 'Ours':
                sys.path.append('third/Ours')
                from third.Ours.lib.xfeat import XFeat

                if subset == 'VIS_SAR':
                    weights = 'third/Ours/models/2024_10_10-10_44_34_VIS_SAR_106000.pth'
                elif subset == 'VIS_IR':
                    weights = 'third/Ours/models/2024_10_21-10_14_50_VIS_IR_62500.pth'
                elif subset == 'VIS_NIR':
                    weights = 'third/Ours/models/2024_11_02-09_43_00_VIS_NIR_30000.pth'
                else:
                    raise ValueError('error image type')
                xfeat = XFeat(weights=weights)
            else:
                xfeat = None

            out_root = 'result/' + method_name + '/' + str(nums_kps) + '/' + subset + '/'
            os.makedirs(out_root, exist_ok=True)

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

            subset_path = os.path.join(SCRIPT_DIR, 'data', subset)
            filepath1 = os.path.join(subset_path, 'test', subset.split('_')[0])
            filepath2 = os.path.join(subset_path, 'test', subset.split('_')[1])
            image_list = sorted(os.listdir(filepath1))
            img_list_whitelist = []
            progress_bar = tqdm(range(len(image_list)))

            img_nums = 0
            all_NCM = []
            all_ME = []
            all_RMSE = []
            all_failed = []
            SR = 0
            for id in progress_bar:
                if subset == 'VIS_NIR' and image_list[id] in blacklist_NIR:
                    continue
                img_nums += 1
                imgpath1 = os.path.join(filepath1, image_list[id])
                imgpath2 = os.path.join(filepath2, image_list[id])

                # Predict matches
                # matches, kp1, kp2 = matcher.match_method(method_name, imgpath1, imgpath2, nums_kps, subset)
                try:
                    t0 = time.time()
                    matches, kp1, kp2 = matcher.match_method(method_name, imgpath1, imgpath2, nums_kps, subset, xfeat)
                    match_time.append(time.time() - t0)
                    if not isinstance(matches, np.ndarray):
                        matches = matches.cpu().numpy()
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

                        # if sum(inliers) >= 4:
                        #     SR += 1
                        # all_NCM.append(sum(inliers))
                        # inlier_ratio.append(irat)
                        # dists_sa.append(corner_dist)
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

                mask_th_3 = dist <= 3
                ncm = sum(dist <= 3)
                if ncm >= 10:
                    SR += 1
                    all_NCM.append(ncm)

                    me = sum(dist[mask_th_3]) / ncm
                    all_ME.append(me)

                    rmse = np.sqrt(sum((dist[mask_th_3]) ** 2) / ncm)
                    all_RMSE.append(rmse)
                else:
                    all_failed.append(image_list[id])

                inlier_ratio.append(irat)
                dists_sa.append(corner_dist)

                if vis_flag:
                    if len(matches) > 0:
                        try:
                            visual_matching(out_root, subset, image_list[id], imgpath1, imgpath2, matches, H_pred, suffix, dist)
                        except Exception as e:
                            print(e)
                # if img_nums > 2:
                #     break

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

            # print Homography
            correct_sa = np.mean([[float(d <= t) for t in thres] for d in dists_sa], axis=0)
            f.write('#### Homography ####')
            f.write('\n')
            for i in range(len(correct_sa)):
                f.write(str(format(correct_sa[i], ".5f")))
                f.write(' ')

            f.write('\n')

            # print Success Rate
            SR = SR / img_nums
            f.write('#### Success Rate ####')
            f.write('\n')
            f.write(str(format(SR, ".5f")))

            f.write('\n')

            # print Average_Matches
            AM = np.mean(n_matches)
            f.write('#### Average Matches ####')
            f.write('\n')
            f.write(str(format(AM, ".5f")))

            f.write('\n')

            # print NCM
            NCM = np.mean(all_NCM)
            f.write('#### NCM ####')
            f.write('\n')
            f.write(str(format(NCM, ".5f")))

            f.write('\n')

            # print ME
            ME = np.mean(all_ME)
            f.write('#### ME ####')
            f.write('\n')
            f.write(str(format(ME, ".5f")))

            f.write('\n')

            # print NCM
            RMSE = np.mean(all_RMSE)
            f.write('#### RMSE ####')
            f.write('\n')
            f.write(str(format(RMSE, ".5f")))

            f.write('\n')

            # print Average_Time
            AT = np.mean(match_time[1:])
            f.write('#### Average Time ####')
            f.write('\n')
            f.write(str(format(AT, ".5f")))

            f.write('\n')

            # print Failed image name
            f.write('#### Failed Images ####')
            f.write('\n')
            if len(all_failed) > 0:
                f.write('Failed num = ' + str(len(all_failed)))
                f.write('\n')
                for i in range(len(all_failed)):
                    f.write(all_failed[i])
                    f.write(',  ')
            else:
                f.write('All Success')

            f.close()



