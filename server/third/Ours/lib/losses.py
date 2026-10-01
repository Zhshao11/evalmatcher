import torch
import torch.nn.functional as F
import torch.nn as nn
import kornia.geometry.transform as KGT
import kornia.filters as KF
import kornia.utils as KU
from third.Ours.lib import utils


def dual_softmax_loss(X, Y, temp=0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp(conf_matrix12).max(dim=-1)[0]
        conf21 = torch.exp(conf_matrix21).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device = X.device)

    loss = F.nll_loss(conf_matrix12, target) + F.nll_loss(conf_matrix21, target)
    return loss, conf


def smooth_l1_loss(input, target, beta=2.0, size_average=True):
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean() if size_average else loss.sum()


def fine_loss(f1, f2, pts1, pts2, fine_module, ws=7):
    '''
        Compute Fine features and spatial loss
    '''
    C, H, W = f1.shape
    N = len(pts1)

    #Sort random offsets
    with torch.no_grad():
        a = -(ws//2)
        b = (ws//2)
        offset_gt = (a - b) * torch.rand(N, 2, device = f1.device) + b
        pts2_random = pts2 + offset_gt

    #pdb.set_trace()
    patches1 = utils.crop_patches(f1.unsqueeze(0), (pts1+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0) #[N, ws*ws, C]
    patches2 = utils.crop_patches(f2.unsqueeze(0), (pts2_random+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0)  #[N, ws*ws, C]

    #Apply transformer
    patches1, patches2 = fine_module(patches1, patches2)

    features = patches1.view(N, ws, ws, C)[:, ws//2, ws//2, :].view(N, 1, 1, C) # [N, 1, 1, C]
    patches2 = patches2.view(N, ws, ws, C) # [N, w, w, C]

    #Dot Product
    heatmap_match = (features * patches2).sum(-1)
    offset_coords = utils.subpix_softmax2d(heatmap_match)

    #Invert offset because center crop inverts it
    offset_gt = -offset_gt 

    #MSE
    error = ((offset_coords - offset_gt)**2).sum(-1).mean()

    #error = smooth_l1_loss(offset_coords, offset_gt)

    return error


def keypoint_position_loss(kpts1, kpts2, pts1, pts2, softmax_temp = 1.0):
    '''
        Computes coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
        for correct offsets
    '''
    C, H, W = kpts1.shape
    kpts1 = kpts1.permute(1,2,0) * softmax_temp
    kpts2 = kpts2.permute(1,2,0) * softmax_temp

    with torch.no_grad():
        #Generate meshgrid
        x, y = torch.meshgrid(torch.arange(W, device=kpts1.device), torch.arange(H, device=kpts1.device), indexing ='xy')
        xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        xy*=8

        #Generate collision map
        hashmap = torch.ones((H*8, W*8, 2), dtype = torch.long, device = kpts1.device) * -1
        hashmap[(pts1[:,1]).long(), (pts1[:,0]).long(), :] = (pts2).long()

        #Estimate offset of src kpts 
        _, kpts1_offsets = kpts1.max(dim=-1)
        kpts1_offsets_x = kpts1_offsets  % 8
        kpts1_offsets_y = kpts1_offsets // 8
        kpts1_offsets_xy = torch.cat([kpts1_offsets_x.unsqueeze(-1), 
                                      kpts1_offsets_y.unsqueeze(-1)], dim=-1)
        #pdb.set_trace()
        kpts1_coords = xy + kpts1_offsets_xy

        #find src -> tgt pts
        kpts1_coords = kpts1_coords.view(-1,2)
        gt_12 = hashmap[kpts1_coords[:,1], kpts1_coords[:,0]]
        mask_valid = torch.all(gt_12 >= 0, dim=-1)
        gt_12 = gt_12[mask_valid]

        #find offset labels
        labels2 = (gt_12/8) - (gt_12/8).long()
        labels2 = (labels2 * 8).long()
        labels2 = labels2[:, 0] + 8*labels2[:, 1] #linear index
        
    kpts2_selected = kpts2[(gt_12[:, 1]/8).long(), (gt_12[:, 0]/8).long()]        

    kpts1_selected = F.log_softmax(kpts1.view(-1,C)[mask_valid], dim=-1)
    kpts2_selected = F.log_softmax(kpts2_selected, dim=-1)

    #Here we enforce softmax to keep current max on src kps
    with torch.no_grad():
        _, labels1 =  kpts1_selected.max(dim=-1)

    predicted2 = kpts2_selected.max(dim=-1)[1]
    acc =  (labels2 == predicted2)
    acc = acc.sum() / len(acc)

    loss = F.nll_loss(kpts1_selected, labels1, reduction = 'mean') + \
           F.nll_loss(kpts2_selected, labels2, reduction = 'mean')
    
    #pdb.set_trace()

    return loss, acc


def coordinate_classification_loss(coords1, pts1, pts2, conf):
    '''
        Computes the fine coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
        for correct offsets after warp
    '''
    #Do not backprop coordinate warps
    with torch.no_grad():

        coords1_detached = pts1 * 1

        #find offset
        offsets1_detached = (coords1_detached/1) - (coords1_detached/1).long()
        offsets1_detached = (offsets1_detached * 1).long()
        labels1 = offsets1_detached[:, 0] + 1*offsets1_detached[:, 1]

    #pdb.set_trace()
    coords1_log = F.log_softmax(coords1, dim=-1)

    predicted = coords1.max(dim=-1)[1]
    acc =  (labels1 == predicted)
    acc = acc[conf > 0.1]
    acc = acc.sum() / len(acc)

    loss = F.nll_loss(coords1_log, labels1, reduction = 'none')
    
    #Weight loss by confidence, giving more emphasis on reliable matches
    conf = conf / conf.sum()
    loss = (loss * conf).sum()

    return loss * 2., acc


def keypoint_loss(heatmap, target):
    # Compute L1 loss
    L1_loss = F.l1_loss(heatmap, target)
    return L1_loss * 3.0


def hard_triplet_loss(X,Y, margin = 0.5):

    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = torch.cdist(X, Y, p=2.0)
    dist_pos = torch.diag(dist_mat)
    dist_neg = dist_mat + 100.*torch.eye(*dist_mat.size(), dtype = dist_mat.dtype, 
            device = dist_mat.get_device() if dist_mat.is_cuda else torch.device("cpu"))

    #filter repeated patches on negative distances to avoid weird stuff on gradients
    dist_neg = dist_neg + dist_neg.le(0.01).float()*100.

    #Margin Ranking Loss
    hard_neg = torch.min(dist_neg, 1)[0]

    loss = torch.clamp(margin + dist_pos - hard_neg, min=0.)

    return loss.mean()


class MMLoss(nn.Module):
    def __init__(self, lam1=1, lam2=1, sample_n=4096, input_size=(192, 192), sample_size=17, safe_radius_neg=7,
                 safe_radius_pos=3, border=5, cuda=True):
        super().__init__()
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)
        self.sample_size = sample_size
        self.sample_n = sample_n
        self.safe_radius_pos = safe_radius_pos
        self.safe_radius_neg = safe_radius_neg
        self.AP = nn.AvgPool2d(sample_size + 1, stride=1, padding=sample_size // 2)
        self.MP = nn.MaxPool2d(sample_size + 1, stride=1, padding=sample_size // 2)
        self.MP3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.MP7 = nn.MaxPool2d(7, stride=1, padding=3)
        self.AP3 = nn.AvgPool2d(3, stride=1, padding=1)
        self.AP5 = nn.AvgPool2d(5, stride=1, padding=2)
        self.border_mask = F.pad(torch.ones([input_size[1] - 2 * border, input_size[0] - 2 * border]),
                                 [border, border, border, border]).unsqueeze(0).unsqueeze(0).long()
        self.running_score_sum = 1000
        self.M_mean = 1000
        self.priori1_mean = 1000
        self.priori2_mean = 1000
        self.mask1_mean = 10
        self.mask2_mean = 10
        self.mask3_mean = 10
        self.running_rep_sum = 100
        self.loss_desc_ = 0
        self.loss_peak_ = 0
        self.loss_rep_ = 0
        if cuda:
            self.AP = self.AP.cuda()
            self.MP = self.MP.cuda()
            self.MP7 = self.MP7.cuda()
            self.MP3 = self.MP3.cuda()
            self.AP3 = self.AP3.cuda()
            self.AP5 = self.AP5.cuda()
            # self.b
            self.border_mask = self.border_mask.cuda().long()

    def extract_patches(self,
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

        sampled = sampled.reshape(ps * ps, -1, c)
        return sampled.permute(1, 0, 2), corner.float()

    def loss_rep(self, score1, score2, pts1, pts2):
        bias = torch.tensor([[self.sample_size // 2] * 2], device=score1.device)

        score1_padded = torch.nn.functional.pad(score1, [self.sample_size // 2] * 4, mode='constant', value=0.)
        idx1 = (pts1 - bias + self.sample_size // 2).int()
        patches1 = self.extract_patches(score1_padded.to(device=idx1.device), idx1, 2 * self.sample_size // 2)[0].unsqueeze(0)
        patches1 = F.normalize(patches1, dim=2).squeeze(3)

        score2_padded = torch.nn.functional.pad(score2, [self.sample_size // 2] * 4, mode='constant', value=0.)
        idx2 = (pts2 - bias + self.sample_size // 2).int()
        patches2 = self.extract_patches(score2_padded.to(device=idx2.device), idx2, 2 * self.sample_size // 2)[0].unsqueeze(0)
        patches2 = F.normalize(patches2, dim=2).squeeze(3)

        cosim = (patches1 * patches2).sum(dim=2, keepdim=True)
        # rep loss weighted with desciptors similairty
        loss_rep = (1.0 - cosim).mean()
        assert not loss_rep.isnan()
        return loss_rep

    def compute_edge(self, im):
        edge = KF.spatial_gradient(im, order=2).abs().sum(dim=[1, 2])
        edge = self.AP3(self.MP7(edge.unsqueeze(1))).detach()
        edge_min = edge.min(dim=-1, keepdim=True)[0]
        edge_min = edge_min.min(dim=-2, keepdim=True)[0]
        edge_max = edge.max(dim=-1, keepdim=True)[0]
        edge_max = edge_max.max(dim=-2, keepdim=True)[0]
        edge = (edge - edge_min) / (edge_max - edge_min)
        return edge

    def loss_peak(self, o_score1, o_score2, im1, im2, conf, pts1, pts2):
        priori1 = self.compute_edge(im1)
        mask1 = 1 - priori1 / (priori1.mean() + 1e-12)
        # mask_pos = mask
        mask1 = F.relu(mask1)
        priori2 = self.compute_edge(im2)
        mask2 = 1 - priori2 / (priori2.mean() + 1e-12)
        mask2 = F.relu(mask2)

        score1 = o_score1 * self.border_mask
        score2 = o_score2 * self.border_mask
        score1_ = KF.gaussian_blur2d(score1, kernel_size=(3, 3), sigma=(1, 1))
        score2_ = KF.gaussian_blur2d(score2, kernel_size=(3, 3), sigma=(1, 1))
        loss_peak_edge = (mask1 * score1.pow(2)).mean() + (mask2 * score2.pow(2)).mean()

        loss_peak_random = self.AP(score1_).pow(2).mean() + (1 - self.MP(score1_)).pow(2).mean() + (self.AP3(score1_) + 1 - self.MP3(score1_)).pow(2).mean() + \
                           self.AP(score2_).pow(2).mean() + (1 - self.MP(score2_)).pow(2).mean() + (self.AP3(score2_) + 1 - self.MP3(score2_)).pow(2).mean()

        p_score1 = o_score1[:, :, pts1[:, 1].long(), pts1[:, 0].long()]
        p_score2 = o_score2[:, :, pts2[:, 1].long(), pts2[:, 0].long()]
        conf = conf / conf.sum()
        loss_peak_coupled = (conf * (1 - p_score1).pow(2)).sum() + (conf * (1 - p_score2).pow(2)).sum()

        loss_peak = (loss_peak_edge + loss_peak_random + loss_peak_coupled)
        # loss_peak = (loss_peak_edge + loss_peak_random)
        assert not loss_peak_edge.isnan()
        assert not loss_peak_random.isnan()
        assert not loss_peak_coupled.isnan()
        assert not loss_peak.isnan()
        return loss_peak


def compute_offset_loss(offset, offset_radius, pred_mask):
    N, C, H, W = pred_mask.shape
    pred_mask = pred_mask.view(N, C, -1)
    pred_mask = F.softmax(pred_mask, dim=-1)
    pred_mask = pred_mask.view(N, C, H, W)
    pred_mask = pred_mask.squeeze(1)

    std = utils.compute_matrix_cov_trace(pred_mask)
    inverse_std = 1. / torch.clamp(std, min=1e-10)
    weight = (inverse_std / torch.mean(inverse_std)).detach()

    x, y = torch.meshgrid(torch.arange(W, device=pred_mask.device), torch.arange(H, device=pred_mask.device),
                        indexing='xy')
    x = x - (W // 2)
    y = y - (H // 2)

    coords_x = (x[None, ...] * pred_mask)
    coords_y = (y[None, ...] * pred_mask)
    coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H * W, 2)
    coords = coords.sum(1)

    dist = torch.abs(offset - coords)
    loss = weight * torch.sqrt(dist[:, 0].pow(2) + dist[:, 1].pow(2))
    loss = loss.mean()
    return loss * 2
    # return loss



