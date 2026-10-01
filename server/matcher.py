import utils
from utils import *
import sys
import cv2
import scipy.io as scio
import torch


def match_method(method, imgpath1, imgpath2, num_kps, subset, xfeat, steerer=None):
    matches, kp1, kp2 = [], [], []
    if method == 'xfeat':
        sys.path.append('third/xfeat')
        from third.xfeat.modules.xfeat import XFeat
        xfeat = XFeat(weights='third/xfeat/weights/xfeat.pt')
        im1 = cv2.imread(imgpath1)
        im2 = cv2.imread(imgpath2)
        matches, kp1, kp2 = xfeat.match_xfeat_star(im1, im2, top_k=num_kps)
        matches = matches[0]

    if method == 'd2net':
        sys.path.append('third/d2net')
        from third.d2net.extract_features import extract
        kp1, des1 = extract(imgpath1, num_kps)
        kp2, des2 = extract(imgpath2, num_kps)

        match_ids, _ = mutual_nn_matching(des1, des2)
        p1s = kp1[match_ids[:, 0], :2]
        p2s = kp2[match_ids[:, 1], :2]
        matches = torch.cat([p1s, p2s], dim=1)

    if method == 'r2d2':
        sys.path.append('third/r2d2')
        from third.r2d2.extract import extract_keypoints
        kp1, des1 = extract_keypoints(imgpath1, num_kps)
        kp2, des2 = extract_keypoints(imgpath2, num_kps)

        match_ids, _ = mutual_nn_matching(des1, des2)
        p1s = kp1[match_ids[:, 0], :2]
        p2s = kp2[match_ids[:, 1], :2]
        matches = torch.cat([p1s, p2s], dim=1)

    if method == 'superglue':
        sys.path.append('third/SuperGlue')
        from third.SuperGlue.match_pairs import superglue_match
        matches, kp1, kp2, _ = superglue_match(imgpath1, imgpath2, num_kps)

    if method == 'loftr':
        sys.path.append('third/LoFTR')
        from third.LoFTR.match_pair import loftr_matcher
        matches, kp1, kp2 = loftr_matcher(imgpath1, imgpath2, num_kps)

    if method == 'dedode':
        sys.path.append('third/DeDoDe')
        from third.DeDoDe.match_pair import dedode_matcher
        matches, kp1, kp2 = dedode_matcher(imgpath1, imgpath2, num_kps)

    if method == 'subpixel':
        sys.path.append('third/keypt2subpx')

    if method == 'alike':
        sys.path.append('third/ALIKE')
        from third.ALIKE.alike_extract import extract
        kp1, des1 = extract(imgpath1, num_kps)
        kp2, des2 = extract(imgpath2, num_kps)

        match_ids, _ = mutual_nn_matching(des1, des2)
        p1s = kp1[match_ids[:, 0], :2]
        p2s = kp2[match_ids[:, 1], :2]
        matches = torch.cat([p1s, p2s], dim=1)

    if method == 'redfeat':
        sys.path.append('third/redfeat')
        from third.redfeat.extract_MMFeat import extract
        kp1, kp2, des1, des2 = extract(imgpath1, imgpath2, num_kps, subset)

        match_ids, _ = mutual_nn_matching(des1, des2)
        p1s = kp1[match_ids[:, 0], :2]
        p2s = kp2[match_ids[:, 1], :2]
        matches = torch.cat([p1s, p2s], dim=1)

    if method == 'sift':
        kp1, des1 = utils.sift_extract(imgpath1, num_kps)
        kp2, des2 = utils.sift_extract(imgpath2, num_kps)

        match_ids, _ = mutual_nn_matching(des1, des2)
        p1s = kp1[match_ids[:, 0], :2]
        p2s = kp2[match_ids[:, 1], :2]
        matches = torch.cat([p1s, p2s], dim=1)

    if method == 'Ours':
        def refine_matches(xfeat, d0, d1, idx0, idx1, steerer_ffeat, rot1to2):
            mkpts_0 = d0['keypoints'][0][idx0]
            mkpts_1 = d1['keypoints'][0][idx1]

            f1_patches = d0['feat_patch'][0][idx0]
            f2_patches = d1['feat_patch'][0][idx1]
            
            if steerer_ffeat:
                while(rot1to2>0):
                    f1_patches = torch.nn.functional.normalize(steerer(f1_patches))
                    rot1to2 = rot1to2 - 1
            pred_mask = xfeat.net.fine_matcher(f1_patches, f2_patches)

            #Compute fine offsets
            offsets = xfeat.subpix_softmax2d(pred_mask.squeeze(1))
            mkpts_0 += offsets

            return mkpts_0.cpu().numpy(), mkpts_1.cpu().numpy()


        def matching(xfeat, steerer, im1, im2, min_cossim=-1, steerer_ffeat=False, top_k=4096):
            with torch.no_grad():
                if steerer == None:
                    matches, kp1, kp2 = xfeat.match_cmodel(im1, im2, top_k=top_k)
                    mkpts_0, mkpts_1 = matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy()
                    rot1to2 = 0
                else:
                    im_set1 = xfeat.norm_input(im1)
                    im_set2 = xfeat.norm_input(im2)

                    # Compute coarse feats
                    out1 = xfeat.detectAndComputeDense(im_set1, top_k=top_k, multiscale=False, type=1)
                    out2 = xfeat.detectAndComputeDense(im_set2, top_k=top_k, multiscale=False, type=2)
                    out1['descriptors'] = out1['descriptors'][0]
                    out2['descriptors'] = out2['descriptors'][0]

                    idxs0, idxs1 = xfeat.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)
                    rot1to2 = 0
                    for r in range(1, 4):
                        # out1['descriptors'] = torch.nn.functional.normalize(steerer(out1['descriptors']), dim=-1)
                        out1['descriptors'] = torch.nn.functional.normalize(steerer(out1['descriptors'].unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2))
                        new_idxs0, new_idxs1 = xfeat.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)
                        if len(new_idxs0) > len(idxs0):
                            idxs0 = new_idxs0
                            idxs1 = new_idxs1
                            rot1to2 = r

                    mkpts_0, mkpts_1 = refine_matches(xfeat, out1, out2, idxs0, idxs1, steerer_ffeat, rot1to2)
                    kp1, kp2 = out1['keypoints'], out2['keypoints']

            return np.hstack((mkpts_0, mkpts_1)), kp1, kp2
        
        im1 = cv2.imread(imgpath1)
        im2 = cv2.imread(imgpath2)
        matches, kp1, kp2 = matching(xfeat, steerer, im1, im2, min_cossim=0.4, steerer_ffeat=True, top_k=num_kps)

    if method == 'SRIF' or method == 'POS-GIFT':
        name = (imgpath1.split('/')[-1]).split('.')[0] + '.mat'
        feature_path = os.path.join('third', method, 'features', name)
        feature = scio.loadmat(feature_path)
        t = feature['features']
        matches = t[0, 0]
        kp1 = t[0, 1]
        kp2 = t[0, 2]

    if method == 'omniglue':
        sys.path.append('third/omniglue')
        from third.omniglue.matcher import omniglue_matcher
        matches = omniglue_matcher(imgpath1, imgpath2, num_kps)

    if method == 'subpx':
        sys.path.append('third/SuperGlue')
        sys.path.append('third/keypt2subpx')
        from third.SuperGlue.match_pairs import superglue_match
        _, kp1, kp2, pred = superglue_match(imgpath1, imgpath2, num_kps)
        from third.keypt2subpx.match_fine import subpx
        matches = subpx(imgpath1, imgpath2, pred)

    if method == 'MINIMA_LG':
        sys.path.append('third/MINIMA')
        from third.MINIMA.match_pairs import minima_match
        method = "sp_lg"
        matches, kp1, kp2 = minima_match(imgpath1, imgpath2, method, num_kps)

    if method == 'MINIMA_LoFTR':
        sys.path.append('third/MINIMA')
        from third.MINIMA.match_pairs import minima_match
        method = "loftr"
        matches, kp1, kp2 = minima_match(imgpath1, imgpath2, method, num_kps)

    if method == 'MINIMA_RoMa':
        sys.path.append('third/MINIMA')
        from third.MINIMA.match_pairs import minima_match
        method = "roma"
        matches, kp1, kp2 = minima_match(imgpath1, imgpath2, method, num_kps)

    if method == 'LoFTR':
        sys.path.append('third/MINIMA')
        from third.MINIMA.match_pairs import minima_match
        method = "LoFTR_outdoor"
        matches, kp1, kp2 = minima_match(imgpath1, imgpath2, method, num_kps)

    if method == 'RoMa':
        sys.path.append('third/MINIMA')
        from third.MINIMA.match_pairs import minima_match
        method = "RoMa_outdoor"
        matches, kp1, kp2 = minima_match(imgpath1, imgpath2, method, num_kps)

    if method == 'MINIMA_XoFTR':
        sys.path.append('third/MINIMA')
        from third.MINIMA.match_pairs import minima_match
        method = "xoftr"
        matches, kp1, kp2 = minima_match(imgpath1, imgpath2, method, num_kps)

    if method == 'XoFTR':
        sys.path.append('third/MINIMA')
        from third.MINIMA.match_pairs import minima_match
        method = "xoftr640"
        matches, kp1, kp2 = minima_match(imgpath1, imgpath2, method, num_kps)

    if method == 'LightGlue':
        sys.path.append('third/LightGlue')
        from third.LightGlue.match_pairs import lightglue_match
        matches, kp1, kp2 = lightglue_match(imgpath1, imgpath2, num_kps)

    return matches, kp1, kp2