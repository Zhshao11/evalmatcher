import argparse
import os
import time
import sys
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from third.Ours.lib.model import *
from third.Ours.lib.augmentation import *
from third.Ours.lib.utils import *
from third.Ours.lib.losses import *
from third.Ours.lib.xfeat import NonMaxSuppression
from torch.utils.data import Dataset, DataLoader


def seed_torch(seed=100):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def parse_arguments():
    parser = argparse.ArgumentParser(description="training script.")
    parser.add_argument('--megadepth_root_path', type=str,
                        default='/media/hmq/humaoqing datasets/deep_matching_data/MegaDepth/phoenix/S6/zl548',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str,
                        default='VIS_SAR/train/',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, default='trained_ckpt',
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='VIS_NIR',
                        choices=['VIS_IR', 'VIS_NIR', 'VIS_SAR'],
                        help='Training scheme. xfeat_default uses both megadepth & synthetic warps.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=160_000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(224, 160), help='Training resolution as width,height. Default is (800, 608). (480, 320)')
    parser.add_argument('--device_num', type=str, default='6',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')
    parser.add_argument('--difficulty', type=float, default=0.05)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args


args = parse_arguments()


class Trainer():
    """
        Class for training XFeat with default params as described in the paper.
        We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
        The major bottleneck is to keep loading huge megadepth h5 files from disk,
        the network training itself is quite fast.
    """

    def __init__(self, difficulty,
                 synthetic_root_path,
                 ckpt_save_path,
                 model_name='xfeat_default',
                 batch_size=10, n_steps=160_000, lr=3e-4, gamma_steplr=0.5,
                 training_res=(800, 608), device_num="0", dry_run=False,
                 save_ckpt_every=500):

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel(fine_model='add').to(self.dev)

        self.NonMaxSuppression = NonMaxSuppression(0.05)

        # Setup optimizer
        self.difficulty = difficulty
        self.offset_radius = 2
        self.sample_n = 4096
        self.batch_size = batch_size
        self.steps = n_steps
        self.training_res = training_res
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        self.augmentor = AugmentationPipe(
            img_dir=synthetic_root_path,
            img_type=model_name,
            device=self.dev, load_dataset=True,
            batch_size=int(self.batch_size * 0.4 if model_name == 'xfeat_default' else batch_size),
            out_resolution=training_res,
            warp_resolution=training_res,
            sides_crop=0.1,
            max_num_imgs=3000,
            num_test_imgs=5,
            photometric=True,
            geometric=True,
            reload_step=4_000
        )
        
        self.start_time = time.strftime("%Y_%m_%d-%H_%M_%S")
        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        os.makedirs(ckpt_save_path + '/model/' + self.start_time, exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path + '/model/' + self.start_time
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + self.start_time)
        self.model_name = model_name

    def train(self):
        self.net.train()
        difficulty = self.difficulty
        Detloss = MMLoss(input_size=self.training_res)
        p1s, p2s, H1, H2 = None, None, None, None
        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if not self.dry_run:
                    if self.augmentor is not None:
                        # Grab synthetic data
                        p1s, p2s, H1, H2, type1, typ2 = make_batch(self.augmentor, difficulty)

                        h_coarse, w_coarse = p1s[0].shape[-2] // 1, p1s[0].shape[-1] // 1
                        _, positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse, sample_n=self.sample_n)

                # Join megadepth & synthetic data
                # norm
                if self.augmentor is not None:
                    p1s = (p1s - p1s.mean(dim=[-1, -2], keepdim=True)) / (p1s.std(dim=[-1, -2], keepdim=True) + 1e-5)
                    p2s = (p2s - p2s.mean(dim=[-1, -2], keepdim=True)) / (p2s.std(dim=[-1, -2], keepdim=True) + 1e-5)

                p1 = p1s
                p2 = p2s
                positives_c = positives_s_coarse

                # Check if batch is corrupted with too few correspondences
                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                # Forward pass
                c_feats1, f_feats1, score1, c_feats2, f_feats2, score2 = self.net(p1, p2)

                loss_items = []

                for b in range(len(positives_c)):
                    # Get positive correspondencies
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                    # Grab features at corresponding idxs
                    m1 = c_feats1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)
                    # pos = torch.cat([pts1[:, 1].unsqueeze(0), pts1[:, 0].unsqueeze(0)], dim=0)
                    # m1, _, idx = interpolate_dense_features(pos, c_feats1[b], return_corners=False)
                    # m1 = m1.permute(1, 0)
                    # assert len(idx) == pos.shape[1]

                    m2 = c_feats2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(1, 0)

                    # Compute losses
                    loss_ds, conf = dual_softmax_loss(m1, m2)

                    margin = self.offset_radius
                    pts1_m, pts2_m, offset = get_coarse_matches(c_feats1[b], c_feats2[b], score1[b], score2[b], H1, H2, b, self.offset_radius, self.NonMaxSuppression, self.augmentor, margin)
                    if len(pts1_m) == 0:
                        loss_offsets = torch.tensor(0).to(self.dev)
                    else:
                        p1_patch, p2_patch = get_cat_patch(f_feats1[b], f_feats2[b], self.offset_radius, pts1_m, pts2_m)
                        pred_mask = self.net.fine_matcher(p1_patch, p2_patch)
                        loss_offsets = compute_offset_loss(offset, self.offset_radius, pred_mask)

                    loss_reps = Detloss.loss_rep(score1[b], score2[b], pts1, pts2)
                    loss_peaks = Detloss.loss_peak(score1[b].unsqueeze(0), score2[b].unsqueeze(0), p1[b].unsqueeze(0), p2[b].unsqueeze(0), conf, pts1, pts2)

                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_offsets.unsqueeze(0))
                    loss_items.append(loss_reps.unsqueeze(0))
                    loss_items.append(loss_peaks.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0, _ = check_accuracy(m1, m2)

                acc_coarse, _ = check_accuracy(m1, m2)

                nb_coarse = len(m1)
                loss = torch.cat(loss_items, -1).mean()
                loss_coarse = loss_ds.item()
                loss_offset = loss_offsets.item()
                loss_peak = loss_peaks.item()
                loss_l1 = loss_reps.item()

                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i + 1) % self.save_ckpt_every == 0 and (i + 1) >= 5000:
                    print('saving iter ', i + 1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i + 1}.pth')

                pbar.set_description(
                    'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} loss_c: {:.3f} loss_f: {:.3f} loss_rep: {:.3f} loss_peak: {:.3f} #matches_c: {:d}'.format(
                        loss.item(), acc_coarse_0, acc_coarse, loss_coarse, loss_offset, loss_l1, loss_peak, nb_coarse))
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                # self.writer.add_scalar('Accuracy/fine_mdepth', acc_coords, i)
                # self.writer.add_scalar('Accuracy/kp_position', acc_pos, i)
                self.writer.add_scalar('Loss/coarse', loss_coarse, i)
                self.writer.add_scalar('Loss/fine', loss_offset, i)
                self.writer.add_scalar('Loss/reliability', loss_l1, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_peak, i)
                self.writer.add_scalar('Count/matches_coarse', nb_coarse, i)


if __name__ == '__main__':
    seed_torch()
    trainer = Trainer(
        difficulty=args.difficulty,
        synthetic_root_path=args.synthetic_root_path,
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every
    )

    # The most fun part
    trainer.train()
