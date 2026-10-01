import os

import cv2
import torch
import numpy as np
import pdb

import viz2d

debug_cnt = -1


def make_batch(augmentor, difficulty = 0.3, train = True):
    l_img_list = augmentor.l_train if train else augmentor.test
    r_img_list = augmentor.r_train if train else augmentor.test

    l_img_type = augmentor.l_type if train else augmentor.test
    r_img_type = augmentor.r_type if train else augmentor.test

    l_batch_images = []
    r_batch_images = []

    l_batch_type = []
    r_batch_type = []

    with torch.no_grad(): # we dont require grads in the augmentation
        for b in range(augmentor.batch_size):
            rdidx = np.random.randint(len(l_img_list))

            l_img = torch.tensor(l_img_list[rdidx], dtype=torch.float32).permute(2,0,1).to(augmentor.device).unsqueeze(0)
            l_batch_images.append(l_img)
            l_batch_type.append(l_img_type[rdidx])

            r_img = torch.tensor(r_img_list[rdidx], dtype=torch.float32).permute(2, 0, 1).to(augmentor.device).unsqueeze(0)
            r_batch_images.append(r_img)
            r_batch_type.append(r_img_type[rdidx])

            assert not l_img_type[rdidx] == r_img_type[rdidx]

        l_batch_images = torch.cat(l_batch_images)
        r_batch_images = torch.cat(r_batch_images)

        p1, H1 = augmentor(l_batch_images, difficulty)
        p2, H2 = augmentor(r_batch_images, difficulty, TPS=True, prob_deformation=0.7)

    return p1, p2, H1, H2, l_batch_type, r_batch_type


def plot_corrs(p1, p2, src_pts, tgt_pts):
    import matplotlib.pyplot as plt
    p1 = p1.cpu()
    p2 = p2.cpu()
    src_pts = src_pts.cpu() ; tgt_pts = tgt_pts.cpu()
    rnd_idx = np.random.randint(len(src_pts), size=200)
    src_pts = src_pts[rnd_idx, ...]
    tgt_pts = tgt_pts[rnd_idx, ...]

    #Plot ground-truth correspondences
    fig, ax = plt.subplots(1,2,figsize=(18, 12))
    colors = np.random.uniform(size=(len(tgt_pts),3))
    #Src image
    img = p1
    for i, p in enumerate(src_pts):
        ax[0].scatter(p[0],p[1],color=colors[i])
    ax[0].imshow(img.permute(1,2,0).numpy()[...,::-1])

    #Target img
    img2 = p2
    for i, p in enumerate(tgt_pts):
        ax[1].scatter(p[0],p[1],color=colors[i])
    ax[1].imshow(img2.permute(1,2,0).numpy()[...,::-1])
    plt.show()


def get_corresponding_pts(p1, p2, H, H2, augmentor, h, w, crop=None, sample_n=4096):
    '''
        Get dense corresponding points
    '''
    global debug_cnt
    negatives, positives = [], []

    with torch.no_grad():
        #real input res of samples
        rh, rw = p1.shape[-2:]
        ratio = torch.tensor([rw/w, rh/h], device = p1.device)

        (H, mask1) = H
        (H2, src, W, A, mask2) = H2

        #Generate meshgrid of target pts
        x, y = torch.meshgrid(torch.arange(w, device=p1.device), torch.arange(h, device=p1.device), indexing ='xy')
        mesh = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        target_pts = mesh.view(-1, 2) * ratio

        #Pack all transformations into T
        for batch_idx in range(len(p1)):
            with torch.no_grad():
                T = (H[batch_idx], H2[batch_idx], 
                    src[batch_idx].unsqueeze(0), W[batch_idx].unsqueeze(0), A[batch_idx].unsqueeze(0))
                #We now warp the target points to src image
                src_pts = (augmentor.get_correspondences(target_pts, T) ) #target to src 
                tgt_pts = (target_pts)
            
                #Check out of bounds points
                mask_valid = (src_pts[:, 0] >=0) & (src_pts[:, 1] >=0) & \
                            (src_pts[:, 0] < rw) & (src_pts[:, 1] < rh)

                negatives.append( tgt_pts[~mask_valid] )            
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]


                #Remove invalid pixels
                mask_valid = mask1[batch_idx, src_pts[:,1].long(), src_pts[:,0].long()] & \
                                mask2[batch_idx, tgt_pts[:,1].long(), tgt_pts[:,0].long()]
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]

                # limit nb of matches if desired
                if crop is not None:
                    rnd_idx = torch.randperm(len(src_pts), device=src_pts.device)[:crop]
                    src_pts = src_pts[rnd_idx]
                    tgt_pts = tgt_pts[rnd_idx]

                if debug_cnt >=0 and debug_cnt < 4:
                    plot_corrs(p1[batch_idx], p2[batch_idx], src_pts , tgt_pts )
                    debug_cnt +=1

                src_pts = (src_pts / ratio)
                tgt_pts = (tgt_pts / ratio)

                #Check out of bounds points
                padto = 10 if crop is not None else 2
                mask_valid1 = (src_pts[:, 0] >= (0 + padto)) & (src_pts[:, 1] >= (0 + padto)) & \
                             (src_pts[:, 0] < (w - padto)) & (src_pts[:, 1] < (h - padto))
                mask_valid2 = (tgt_pts[:, 0] >= (0 + padto)) & (tgt_pts[:, 1] >= (0 + padto)) & \
                             (tgt_pts[:, 0] < (w - padto)) & (tgt_pts[:, 1] < (h - padto))
                mask_valid = mask_valid1 & mask_valid2
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]         

                #Remove repeated correspondences
                lut_mat = torch.ones((h, w, 4), device = src_pts.device, dtype = src_pts.dtype) * -1
                # src_pts_np = src_pts.cpu().numpy()
                # tgt_pts_np = tgt_pts.cpu().numpy()
                try:
                    lut_mat[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
                    mask_valid = torch.all(lut_mat >= 0, dim=-1)
                    points = lut_mat[mask_valid]

                    if len(points) > sample_n:
                        idx = torch.randint(0, len(points), [sample_n])
                        points = points[idx]
                    positives.append(points)
                except:
                    pdb.set_trace()
                    print('..')

    return negatives, positives


def crop_patches(tensor, coords, size = 7):
    '''
        Crop [size x size] patches around 2D coordinates from a tensor.
    '''
    B, C, H, W = tensor.shape

    x, y = coords[:, 0], coords[:, 1]
    y = y.view(-1, 1, 1)
    x = x.view(-1, 1, 1)
    halfsize = size // 2
    # Create meshgrid for indexing
    x_offset, y_offset = torch.meshgrid(torch.arange(-halfsize, halfsize+1), torch.arange(-halfsize, halfsize+1), indexing='xy')
    y_offset = y_offset.to(tensor.device)
    x_offset = x_offset.to(tensor.device)

    # Compute indices around each coordinate
    y_indices = (y + y_offset.view(1, size, size)).squeeze(0) + halfsize
    x_indices = (x + x_offset.view(1, size, size)).squeeze(0) + halfsize

    # Handle out-of-boundary indices with padding
    tensor_padded = torch.nn.functional.pad(tensor, (halfsize, halfsize, halfsize, halfsize), mode='constant')

    # Index tensor to get patches
    patches = tensor_padded[:, :, y_indices, x_indices] # [B, C, N, H, W]
    return patches

def subpix_softmax2d(heatmaps, temp = 0.25):
    N, H, W = heatmaps.shape
    heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
    x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
    x = x - (W//2)
    y = y - (H//2)
    #pdb.set_trace()
    coords_x = (x[None, ...] * heatmaps)
    coords_y = (y[None, ...] * heatmaps)
    coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
    coords = coords.sum(1)

    return coords


def check_accuracy(X, Y, pts1 = None, pts2 = None, plot=False):
    with torch.no_grad():
        #dist_mat = torch.cdist(X,Y)
        dist_mat = X @ Y.t()

        nn1 = torch.argmax(dist_mat, dim=1)
        correct1 = nn1 == torch.arange(len(X), device = X.device)

        nn2 = torch.argmax(dist_mat.permute(1, 0), dim=1)
        correct2 = nn2 == torch.arange(len(Y), device = Y.device)
        
        correct = correct1 * correct2

        if pts1 is not None and plot:
            import matplotlib.pyplot as plt
            canvas = torch.zeros((60, 80),device=X.device)
            pts1 = pts1[~correct]
            canvas[pts1[:,1].long(), pts1[:,0].long()] = 1
            canvas = canvas.cpu().numpy()
            plt.imshow(canvas), plt.show()

        acc = correct.sum().item() / len(X)
        return acc, correct

def get_nb_trainable_params(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	nb_params = sum([np.prod(p.size()) for p in model_parameters])
 
	print('Number of trainable parameters: {:d}'.format(nb_params))


def extract_patches(
    tensor: torch.Tensor,
    required_corners: torch.Tensor,
    ps: int,
) -> torch.Tensor:
    c, h, w = tensor.shape
    corner = required_corners.long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offset = torch.arange(0, ps)

    kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
    x, y = torch.meshgrid(offset, offset, **kw)
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    assert sampled.shape[:3] == patches.shape[:3]
    return sampled.permute(2, 3, 0, 1), corner.float()


def get_cat_patch(feats1, feats2, offset_radius, pts1, pts2):
    # with torch.no_grad():

    bias = torch.tensor([[offset_radius] * 2], device=feats1.device)
    idx = (pts1 - bias + offset_radius).int()
    feats1_padded = torch.nn.functional.pad(feats1, [offset_radius] * 4, mode='constant', value=0.)
    p1_patch = extract_patches(feats1_padded.to(device=idx.device), idx, 2 * offset_radius + 1)[0]

    p2_patch = feats2[:, pts2[:, 1].long(), pts2[:, 0].long()]
    # pos = torch.cat([pts1[:, 1].unsqueeze(0), pts1[:, 0].unsqueeze(0)], dim=0)
    # p1_patch, _, idx = interpolate_dense_features(pos, feats1, return_corners=False)
    p2_patch = p2_patch.permute(1, 0).unsqueeze(2).unsqueeze(3)
    p2_patch = p2_patch.expand(-1, -1, offset_radius * 2 + 1, offset_radius * 2 + 1)
    # assert len(idx) == pos.shape[1]

    return p1_patch, p2_patch


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


def visual_matching(method_name, subset, img_name, im1, im2, matches, H_pred, suffix):
    viz_path = 'result/' + method_name + '/viz/' + subset
    os.makedirs(viz_path, exist_ok=True)

    matches_path = os.path.join(viz_path, 'matches')
    os.makedirs(matches_path, exist_ok=True)
    fusion_path = os.path.join(viz_path, 'fusion')
    os.makedirs(fusion_path, exist_ok=True)
    board_path = os.path.join(viz_path, 'board')
    os.makedirs(board_path, exist_ok=True)

    # matching
    viz2d.plot_images([np.asarray(im1), np.asarray(im2)])
    viz2d.plot_matches(matches[:, :2], matches[:, 2:], color='lime', lw=0.2)
    viz2d.save_plot(os.path.join(matches_path, img_name))

    # fusion_board
    if suffix == '.12':
        image_fusion(np.asarray(im1), np.asarray(im2), H_pred, os.path.join(fusion_path, img_name), os.path.join(board_path, img_name))
    else:
        image_fusion(np.asarray(im2), np.asarray(im1), H_pred, os.path.join(fusion_path, img_name), os.path.join(board_path, img_name))


def interpolate_dense_features(pos, dense_features, return_corners=False):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    _, h, w = dense_features.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise ValueError('ids none')

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    descriptors = (
            w_top_left * dense_features[:, i_top_left, j_top_left] +
            w_top_right * dense_features[:, i_top_right, j_top_right] +
            w_bottom_left * dense_features[:, i_bottom_left, j_bottom_left] +
            w_bottom_right * dense_features[:, i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    if not return_corners:
        return [descriptors, pos, ids]
    else:
        corners = torch.stack([
            torch.stack([i_top_left, j_top_left], dim=0),
            torch.stack([i_top_right, j_top_right], dim=0),
            torch.stack([i_bottom_left, j_bottom_left], dim=0),
            torch.stack([i_bottom_right, j_bottom_right], dim=0)
        ], dim=0)
        return [descriptors, pos, ids, corners]
    

def compute_matrix_cov_trace(x):
    with torch.no_grad():
        mean_per_matrix = x.mean(dim=(1, 2), keepdim=True)  # [4096, 1, 1]  
  
        # 计算每个矩阵中心化后的平方和，即方差的和（未除以 n-1，因为这里只计算迹）  
        centered_x = x - mean_per_matrix  
        squared_centered_x = centered_x ** 2  
        variance_sum_per_matrix = squared_centered_x.sum(dim=(1, 2))  # [4096]  
        
        # 由于协方差矩阵的迹等于方差的和，所以我们直接输出结果  
        # output = variance_sum_per_matrix.view(4096, 1)  # [4096, 1]  

        return variance_sum_per_matrix


def get_des_kp(des, score, NonMaxSuppression, top_k=4096):
    y, x = NonMaxSuppression(score)
    q = score[0, 0, y, x]
    if len(q) < top_k:
        idxs = q.topk(len(q))[1]
    else:
        idxs = q.topk(top_k)[1]
    y = y[idxs]
    x = x[idxs]

    feats = des[0, :, y, x].t()
    mkpts = torch.stack([x, y], dim=-1)
    return feats, mkpts


def batch_match(feats1, feats2, min_cossim = -1):
		B = len(feats1)
		cossim = torch.bmm(feats1, feats2.permute(0,2,1))
		match12 = torch.argmax(cossim, dim=-1)
		match21 = torch.argmax(cossim.permute(0,2,1), dim=-1)

		idx0 = torch.arange(len(match12[0]), device=match12.device)

		batched_matches = []

		for b in range(B):
			mutual = match21[b][match12[b]] == idx0

			if min_cossim > 0:
				cossim_max, _ = cossim[b].max(dim=1)
				good = cossim_max > min_cossim
				idx0_b = idx0[mutual & good]
				idx1_b = match12[b][mutual & good]
			else:
				idx0_b = idx0[mutual]
				idx1_b = match12[b][mutual]

			batched_matches.append((idx0_b, idx1_b))

		return batched_matches


def get_coarse_matches(des1, des2, score1, score2, H, H2, batch_idx, offset_radius, NonMaxSuppression, augmentor, margin):
    with torch.no_grad():
        feat1, kp1 = get_des_kp(des1.unsqueeze(0), score1.unsqueeze(0), NonMaxSuppression)
        feat2, kp2 = get_des_kp(des2.unsqueeze(0), score2.unsqueeze(0), NonMaxSuppression)

        if len(feat1) == 0 or len(feat2) == 0 :
            return [], [], []
        else:
            batched_matches = batch_match(feat1.unsqueeze(0), feat2.unsqueeze(0))
            idx1, idx2 = batched_matches[0]
            mkpts_1 = kp1[idx1]
            mkpts_2 = kp2[idx2]

            (H, mask1) = H
            (H2, src, W, A, mask2) = H2
            T = (H[batch_idx], H2[batch_idx], src[batch_idx].unsqueeze(0), W[batch_idx].unsqueeze(0), A[batch_idx].unsqueeze(0))
            gt_mkpts_1 = (augmentor.get_correspondences(mkpts_2, T) ) #target to src
            offset = gt_mkpts_1 - mkpts_1

            mask_valid = (offset[:, 0] >= -margin) & (offset[:, 1] >= -margin) & (offset[:, 0] <= margin) & (offset[:, 1] <= margin)
            offset = offset[mask_valid]
            mkpts_1 = mkpts_1[mask_valid]
            mkpts_2 = mkpts_2[mask_valid]

            return mkpts_1, mkpts_2, offset