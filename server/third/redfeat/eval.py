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
import viz2d
from match_fusion import redfeat


def cal_reproj_dists_H(p1s, p2s, homography):
    '''Compute the reprojection errors using the GT homography'''
    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)  # Homogenous
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist


def checkboard(im1, im2, d=150):
    im1 = im1 * 1.0
    im2 = im2 * 1.0
    mask = np.zeros_like(im1)
    for i in range(mask.shape[0] // d + 1):
        for j in range(mask.shape[1] // d + 1):
            if (i + j) % 2 == 0:
                mask[i * d:(i + 1) * d, j * d:(j + 1) * d, :] += 1
    return im1 * mask + im2 * (1 - mask)


def image_fusion(img1_np, img2_np, solution, f_path, b_path):
    img1_np = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
    img2_np = cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)

    M1, N1, num1 = img1_np.shape
    M2, N2, num2 = img2_np.shape

    # Create a blank fusion image
    if num1 == 3 and num2 == 3:
        fusion_image = np.zeros((3 * M1, 3 * N1, num1), dtype=np.uint8)
    elif num1 == 1 and num2 == 3:
        fusion_image = np.zeros((3 * M1, 3 * N1), dtype=np.uint8)
        img2_np = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
    elif num1 == 3 and num2 == 1:
        fusion_image = np.zeros((3 * M1, 3 * N1), dtype=np.uint8)
        img1_np = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    elif num1 == 1 and num2 == 1:
        fusion_image = np.zeros((3 * M1, 3 * N1), dtype=np.uint8)

    # Create an identity transformation matrix
    solution_1 = np.array([[1, 0, N1], [0, 1, M1], [0, 0, 1]], dtype=np.float32)

    # Apply the transformation to the first image
    f_1 = cv2.warpPerspective(img1_np, solution_1, (3 * N1, 3 * M1))

    # Apply the transformation to the second image using the provided solution
    f_2 = cv2.warpPerspective(img2_np, solution_1 @ solution, (3 * N1, 3 * M1))

    # Find overlapping regions and blend images
    same_index = np.where((f_1 != 0) & (f_2 != 0))  # 相同区域
    index_1 = np.where((f_1 != 0) & (f_2 == 0))  # 在 f_1 中而不在 f_2 中的区域
    index_2 = np.where((f_1 == 0) & (f_2 != 0))  # 在 f_2 中而不在 f_1 中的区域

    fusion_image[same_index] = f_1[same_index] // 2 + f_2[same_index] // 2
    fusion_image[index_1] = f_1[index_1]
    fusion_image[index_2] = f_2[index_2]

    fusion_image = fusion_image.astype(np.uint8)

    # Delete redundant areas
    left_up = np.dot(solution_1 @ solution, [1, 1, 1])
    left_down = np.dot(solution_1 @ solution, [1, M2, 1])
    right_up = np.dot(solution_1 @ solution, [N2, 1, 1])
    right_down = np.dot(solution_1 @ solution, [N2, M2, 1])

    X = [left_up[0] / left_up[2], left_down[0] / left_down[2], right_up[0] / right_up[2], right_down[0] / right_down[2]]
    Y = [left_up[1] / left_up[2], left_down[1] / left_down[2], right_up[1] / right_up[2], right_down[1] / right_down[2]]

    X_min = max(int(np.floor(min(X))), 1)
    X_max = min(int(np.ceil(max(X))), 3 * N1)
    Y_min = max(int(np.floor(min(Y))), 1)
    Y_max = min(int(np.ceil(max(Y))), 3 * M1)

    if X_min > N1 + 1:
        X_min = N1 + 1
    if X_max < 2 * N1:
        X_max = 2 * N1
    if Y_min > M1 + 1:
        Y_min = M1 + 1
    if Y_max < 2 * M1:
        Y_max = 2 * M1

    if num1 == 1:
        fusion_image = fusion_image[Y_min:Y_max, X_min:X_max]
        f_1 = f_1[Y_min:Y_max, X_min:X_max]
        f_2 = f_2[Y_min:Y_max, X_min:X_max]
    elif num1 == 3:
        fusion_image = fusion_image[Y_min:Y_max, X_min:X_max, :]
        f_1 = f_1[Y_min:Y_max, X_min:X_max, :]
        f_2 = f_2[Y_min:Y_max, X_min:X_max, :]

    # save the fusion image
    cv2.imwrite(f_path, fusion_image)

    grid_num = 5  # board nun
    grid_size = min(f_1.shape[0], f_1.shape[1]) // grid_num  # board size

    f_3 = checkboard(f_1, f_2, grid_size)
    # save the board image
    cv2.imwrite(b_path, f_3)


def visual_matching(method_name, subset, img_name, im1, im2, matches, H_pred, suffix, dist):
    viz_path = 'result/' + method_name + '/viz/' + subset
    os.makedirs(viz_path, exist_ok=True)

    matches_path = os.path.join(viz_path, 'matches')
    os.makedirs(matches_path, exist_ok=True)
    fusion_path = os.path.join(viz_path, 'fusion')
    os.makedirs(fusion_path, exist_ok=True)
    board_path = os.path.join(viz_path, 'board')
    os.makedirs(board_path, exist_ok=True)

    color = []
    for i in range(len(dist)):
        if dist[i] <= 5:
            t = [0, 1, 0]
        else:
            t = [1, 0, 0]
        color.append(t)

    # matching
    viz2d.plot_images([np.asarray(im1), np.asarray(im2)])
    # viz2d.plot_matches(matches[:, :2], matches[:, 2:], color='lime', lw=0.2)
    viz2d.plot_matches(matches[:, :2], matches[:, 2:], color=color, lw=0.2)
    viz2d.save_plot(os.path.join(matches_path, img_name))

    # fusion_board
    if suffix == '.12':
        image_fusion(np.asarray(im1), np.asarray(im2), H_pred, os.path.join(fusion_path, img_name), os.path.join(board_path, img_name))
    else:
        image_fusion(np.asarray(im2), np.asarray(im1), H_pred, os.path.join(fusion_path, img_name), os.path.join(board_path, img_name))


method_name = 'redfeat'
# blacklist_NIR = ['89.png', '87.png', '105.png', '129.png']
subset = 'VIS_IR'
nums_kp = 4096
vis_flag = True
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
all_ME = []
all_RMSE = []
all_failed = []
SR = 0
for id in progress_bar:
    img_nums += 1

    # if subset == 'NIR' and image_list[id] in blacklist_NIR:
    #     continue
    # else:
    #     img_list_whitelist.append(image_list[id])
    imgpath1 = os.path.join(filepath1, image_list[id])
    imgpath2 = os.path.join(filepath2, image_list[id])

    im1 = cv2.imread(imgpath1)
    im2 = cv2.imread(imgpath2)

    # Predict matches
    try:
        t0 = time.time()
        matches = redfeat(imgpath1, imgpath2, 'Pretrained/VIS_IR.pth')
        match_time.append(time.time() - t0)
        # matches = matches[0].cpu().numpy()
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
        if len(inliers) > 0:
            visual_matching(method_name, subset, image_list[id], im1, im2, matches, H_pred, suffix, dist)
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



