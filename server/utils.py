import os

import cv2
import torch
import numpy as np
import pdb
import viz2d


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


def visual_matching(out_root, subset, img_name, imgpath1, imgpath2, matches, H_pred, suffix, dist):
    im1 = cv2.imread(imgpath1)
    im2 = cv2.imread(imgpath2)
    viz_path = out_root + 'viz/'
    os.makedirs(viz_path, exist_ok=True)

    matches_path = os.path.join(viz_path, 'matches')
    os.makedirs(matches_path, exist_ok=True)
    fusion_path = os.path.join(viz_path, 'fusion')
    os.makedirs(fusion_path, exist_ok=True)
    board_path = os.path.join(viz_path, 'board')
    os.makedirs(board_path, exist_ok=True)
    inliers_path = os.path.join(viz_path, 'inliers')
    os.makedirs(inliers_path, exist_ok=True)

    ts = 3
    color = []
    for i in range(len(dist)):
        if dist[i] <= ts:
            t = [0, 1, 0]
        else:
            t = [1, 0, 0]
        color.append(t)

    # all matching
    viz2d.plot_images([np.asarray(im1), np.asarray(im2)])
    viz2d.plot_matches(matches[:, :2], matches[:, 2:], color=color, lw=0.2)
    # viz2d.plot_matches(matches[:, :2], matches[:, 2:], color='lime', lw=0.2)
    viz2d.save_plot(os.path.join(matches_path, img_name))

    # inliers matching
    mask = dist <= ts
    inliers = matches[mask]
    if len(inliers) > 0:
        viz2d.plot_images([np.asarray(im1), np.asarray(im2)])
        viz2d.plot_matches(inliers[:, :2], inliers[:, 2:], color='lime', lw=0.2)
        viz2d.save_plot(os.path.join(inliers_path, img_name))

    # fusion_board
    if suffix == '.12':
        image_fusion(np.asarray(im1), np.asarray(im2), H_pred, os.path.join(fusion_path, img_name), os.path.join(board_path, img_name))
    else:
        image_fusion(np.asarray(im2), np.asarray(im1), H_pred, os.path.join(fusion_path, img_name), os.path.join(board_path, img_name))


def mutual_nn_matching_torch(desc1, desc2, threshold=None, eps=1e-9):
    if len(desc1) == 0 or len(desc2) == 0:
        return torch.empty((0, 2), dtype=torch.int64), torch.empty((0, 2), dtype=torch.int64)

    device = desc1.device
    # desc1 = desc1 / (desc1.norm(dim=1, keepdim=True) + eps)
    # desc2 = desc2 / (desc2.norm(dim=1, keepdim=True) + eps)
    similarity = torch.einsum('id, jd->ij', desc1, desc2)

    nn12 = similarity.max(dim=1)[1]
    nn21 = similarity.max(dim=0)[1]
    ids1 = torch.arange(0, similarity.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    scores = similarity.max(dim=1)[0][mask]
    if threshold:
        mask = scores > threshold
        matches = matches[mask]
        scores = scores[mask]
    return matches, scores


def mutual_nn_matching(desc1, desc2, threshold=None):
    if isinstance(desc1, np.ndarray):
        desc1 = torch.from_numpy(desc1)
        desc2 = torch.from_numpy(desc2)
    matches, scores = mutual_nn_matching_torch(desc1, desc2, threshold=threshold)
    return matches.cpu().numpy(), scores.cpu().numpy()


def sift_extract(imgpath, num_kps):
    from copy import deepcopy
    img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=-10000, edgeThreshold=-10000)
    keypoints, scales, angles, responses = get_SIFT_keypoints(sift, img)
    kpts = [cv2.KeyPoint(x=keypoints[i][0], y=keypoints[i][1], size=scales[i], angle=angles[i]) for i in range(min(num_kps, len(scales)))]
    desc = sift.compute(img, kpts)[1] * 1.0
    desc = desc / np.linalg.norm(desc, axis=1, keepdims=True)
    kp = deepcopy(keypoints[0:min(num_kps, len(scales))])

    return torch.from_numpy(kp).cuda(), torch.from_numpy(desc).cuda()


def get_SIFT_keypoints(sift, img, tic=False):
    import torch.nn as nn
    AP = nn.AvgPool2d(9, stride=1, padding=4)
    MP = nn.AvgPool2d(9, stride=1, padding=4)
    # convert to gray-scale and compute SIFT keypoints
    keypoints = sift.detect(img, None)
    # keypoints = sift.detect(img, None)
    img_tensor = torch.from_numpy(img*1.0/255).permute(2,0,1).unsqueeze(0)
    mask_extra = (MP((img_tensor>1e-12).sum(dim=1,keepdim=True).float())>1e-5).float()
    for ii in range(2):
        mask_extra = (AP(mask_extra)>0.9999).float()
    mask_extra = mask_extra.squeeze(0).squeeze(0).numpy()
    response = []

    for kp in keypoints:
        x,y = kp.pt
        x = min(int(x+0.5),mask_extra.shape[0]-1)
        y = min(int(y+0.5),mask_extra.shape[1]-1)
        t = mask_extra[y,x]
        response.append(t*kp.response)
    #response = np.array([kp.response for kp in keypoints])
    respSort = np.argsort(response)[::-1]

    pt = np.array([kp.pt for kp in keypoints])[respSort]
    size = np.array([kp.size for kp in keypoints])[respSort]
    angle = np.array([kp.angle for kp in keypoints])[respSort]
    response = np.array([kp.response for kp in keypoints])[respSort]
    #print(time_)
    if tic:
        return pt, size, angle, response
    return pt, size, angle, response