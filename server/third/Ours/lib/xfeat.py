
"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import numpy as np
import os

import pydegensac
import torch
import torch.nn.functional as F

import tqdm

from third.Ours.lib.model import *
from third.Ours.lib.interpolator import InterpolateSparse2d
from third.Ours.lib.utils import extract_patches


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rep_thr=0.001):
        super(NonMaxSuppression, self).__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 3,1,1
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


class XFeat(nn.Module):
	""" 
		Implements the inference module for XFeat. 
		It supports inference for both sparse and semi-dense feature extraction & matching.
	"""

	def __init__(self, weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat.pt', top_k = 4096, detection_threshold=0.001):
		super().__init__()
		self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.net = XFeatModel().to(self.dev).eval()
		self.top_k = top_k
		self.detection_threshold = detection_threshold
		self.NonMaxSuppression = NonMaxSuppression(detection_threshold)

		if weights is not None:
			if isinstance(weights, str):
				# print('loading weights from: ' + weights)
				self.net.load_state_dict(torch.load(weights, map_location=self.dev))
			else:
				self.net.load_state_dict(weights)

		self.interpolator = InterpolateSparse2d('bicubic')

		#Try to import LightGlue from Kornia
		self.kornia_available = False
		self.lighterglue = None
		try:
			import kornia
			self.kornia_available=True
		except:
			pass

	@torch.inference_mode()
	def detectAndComputeDense(self, x, top_k=None, multiscale=False, type=1):
		"""
			Compute dense *and coarse* descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return: features sorted by their reliability score -- from most to least
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
					'scales'       ->   torch.Tensor(top_k,): extraction scale
					'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
		"""
		if top_k is None: top_k = self.top_k
		if multiscale:
			mkpts, sc, feats, feat_patch = self.extract_dualscale(x, top_k, type)
		else:
			mkpts, feats, feat_patch, q = self.extractDense(x, top_k, type)
			sc = torch.ones(mkpts.shape[:2], device=mkpts.device)

		return {'keypoints': mkpts,
				'descriptors': feats,
				'scales': sc,
				'feat_patch': feat_patch}

	@torch.inference_mode()
	def match_cmodel(self, im_set1, im_set2, top_k=None):
		"""
            Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
            input:
                im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                top_k -> int: keep best k features
            returns:
                matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
        """
		if top_k is None: top_k = self.top_k
		im_set1 = self.norm_input(im_set1)
		im_set2 = self.norm_input(im_set2)

		# Compute coarse feats
		out1 = self.detectAndComputeDense(im_set1, top_k=top_k, multiscale=False, type=1)
		out2 = self.detectAndComputeDense(im_set2, top_k=top_k, multiscale=False, type=2)

		# Match batches of pairs
		idxs_list = self.batch_match(out1['descriptors'], out2['descriptors'], min_cossim=-1)
		B = len(im_set1)

		# Refine coarse matches
		# this part is harder to batch, currently iterate
		matches = []
		for b in range(B):
			matches.append(self.refine_matches(out1, out2, matches=idxs_list, batch_idx=b))

		return matches, out1['keypoints'], out2['keypoints']

	def preprocess_tensor(self, x):
		""" Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
		if isinstance(x, np.ndarray) and len(x.shape) == 3:
			x = torch.tensor(x).permute(2,0,1)[None]
		x = x.to(self.dev).float()

		H, W = x.shape[-2:]
		_H, _W = (H//32) * 32, (W//32) * 32
		rh, rw = H/_H, W/_W

		x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
		return x, rh, rw

	@torch.inference_mode()
	def batch_match(self, feats1, feats2, min_cossim = -1):
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

	def subpix_softmax2d(self, heatmaps):
		N, H, W = heatmaps.shape
		heatmaps = torch.softmax(heatmaps.view(-1, H*W), -1).view(-1, H, W)
		x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
		x = x - (W//2)
		y = y - (H//2)

		coords_x = (x[None, ...] * heatmaps)
		coords_y = (y[None, ...] * heatmaps)
		coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
		coords = coords.sum(1)

		return coords

	def refine_matches(self, d0, d1, matches, batch_idx, fine_conf=0.25):
		idx0, idx1 = matches[batch_idx]
		feats1 = d0['descriptors'][batch_idx][idx0]
		feats2 = d1['descriptors'][batch_idx][idx1]
		mkpts_0 = d0['keypoints'][batch_idx][idx0]
		mkpts_1 = d1['keypoints'][batch_idx][idx1]
		sc0 = d0['scales'][batch_idx][idx0]

		# H_pred, inliers = pydegensac.findHomography(mkpts_0.cpu().numpy(), mkpts_1.cpu().numpy(), 1)

		f1_patches = d0['feat_patch'][batch_idx][idx0]
		f2_patches = d1['feat_patch'][batch_idx][idx1]
		# cat_patch = torch.cat([f2_patches, f1_patches], dim=1)
		pred_mask = self.net.fine_matcher(f1_patches, f2_patches)

		#Compute fine offsets
		offsets = self.subpix_softmax2d(pred_mask.squeeze(1))
		mkpts_0 += offsets

		return torch.cat([mkpts_0, mkpts_1], dim=-1)

	@torch.inference_mode()
	def match(self, feats1, feats2, min_cossim = 0.82):

		cossim = feats1 @ feats2.t()
		cossim_t = feats2 @ feats1.t()
		
		_, match12 = cossim.max(dim=1)
		_, match21 = cossim_t.max(dim=1)

		idx0 = torch.arange(len(match12), device=match12.device)
		mutual = match21[match12] == idx0

		if min_cossim > 0:
			cossim, _ = cossim.max(dim=1)
			good = cossim > min_cossim
			idx0 = idx0[mutual & good]
			idx1 = match12[mutual & good]
		else:
			idx0 = idx0[mutual]
			idx1 = match12[mutual]

		return idx0, idx1

	def extractDense(self, im, top_k=8_000, type=1, offset_radius=2):
		if top_k < 2048:
			top_k += 300

		im, rh1, rw1 = self.preprocess_tensor(im)

		if type== 1:
			c_des, f_des, score = self.net.forward1(im)
		else:
			c_des, f_des, score = self.net.forward2(im)

		y, x = self.NonMaxSuppression(score)
		q = score[0, 0, y, x]
		if len(q) < top_k:
			idxs = q.topk(len(q))[1]
		else:
			idxs = q.topk(top_k)[1]

		q = q[idxs]
		y = y[idxs]
		x = x[idxs]

		c_feats = c_des[0, :, y, x].t()
		mkpts = torch.stack([x, y], dim=-1)

		if type == 2:
			f_feats = f_des[0, :, y, x].t()
			f_feat_patch = f_feats.unsqueeze(2).unsqueeze(3)
			f_feat_patch = f_feat_patch.expand(-1, -1, offset_radius * 2 + 1, offset_radius * 2 + 1)
		else:
			bias = torch.tensor([[offset_radius] * 2], device=f_des.device)
			idx = (mkpts - bias + offset_radius).int()

			f_des_padded = torch.nn.functional.pad(f_des.squeeze(0), [offset_radius] * 4, mode='constant', value=0.)
			f_des_patch = extract_patches(f_des_padded.to(device=f_des.device), idx, 2 * offset_radius + 1)[0]
			# im_padded = torch.nn.functional.pad(im.squeeze(0), [offset_radius] * 4, mode='constant', value=0.)
			# im_patch = extract_patches(im_padded.to(device=im.device), idx, 2 * offset_radius + 1)[0]

			# feat_patch = torch.cat([des_patch, im_patch], dim=1)
			f_feat_patch = f_des_patch

		mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1,-1)

		return mkpts.unsqueeze(0), c_feats.unsqueeze(0), f_feat_patch.unsqueeze(0), q.unsqueeze(0)

	def extract_dualscale(self, x, top_k, type=1, s1=0.6, s2=1.3):
		x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
		x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')

		B, _, _, _ = x.shape

		# mkpts_1, feats_1, feat_patch_1, q_1 = self.extractDense(x1, int(top_k*0.20), type)
		# mkpts_2, feats_2, feat_patch_2, q_2 = self.extractDense(x2, int(top_k*0.80), type)
		mkpts_1, feats_1, feat_patch_1, q_1 = self.extractDense(x1, top_k, type)
		mkpts_2, feats_2, feat_patch_2, q_2 = self.extractDense(x2, top_k, type)

		mkpts = torch.cat([mkpts_1/s1, mkpts_2/s2], dim=1)
		sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1/s1)
		sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1/s2)
		sc = torch.cat([sc1, sc2],dim=1)
		feats = torch.cat([feats_1, feats_2], dim=1)
		feat_patch = torch.cat([feat_patch_1, feat_patch_2], dim=1)

		q = torch.cat([q_1, q_2], dim=1).squeeze(0)
		if len(q) < top_k:
			idxs = q.topk(len(q))[1]
		else:
			idxs = q.topk(top_k)[1]
		mkpts = mkpts[:, idxs]
		sc = sc[:, idxs]
		feats = feats[:, idxs]
		feat_patch = feat_patch[:, idxs]
		return mkpts, sc, feats, feat_patch

	def norm_input(self, x):
		if len(x.shape) == 3:
			x = x[None, ...]

		if isinstance(x, np.ndarray):
			x = torch.tensor(x).permute(0,3,1,2)/255

		x = (x - x.mean(dim=[-1, -2], keepdim=True)) / x.std(dim=[-1, -2], keepdim=True)
		return x
