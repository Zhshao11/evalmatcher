import argparse
import numpy as np
from load_model import load_model


def minima_match(imgpath1, imgpath2, method, num_kps):
    def add_common_arguments(parser):
        parser.add_argument('--exp_name', type=str, default="VisSYN")
        parser.add_argument('--fig1', type=str, default="./demo/vis_test.png")
        parser.add_argument('--fig2', type=str, default="./demo/depth_test.png")
        parser.add_argument('--save_dir', type=str, default="./demo/")

    def add_method_arguments(parser, method):
        if method == "xoftr":
            parser.add_argument('--match_threshold', type=float, default=0.3)
            parser.add_argument('--fine_threshold', type=float, default=0.1)
            parser.add_argument('--ckpt', type=str, default="./third/MINIMA/weights/minima_xoftr.ckpt")

        elif method == "xoftr640":
            parser.add_argument('--match_threshold', type=float, default=0.3)
            parser.add_argument('--fine_threshold', type=float, default=0.1)
            parser.add_argument('--ckpt', type=str, default="./third/MINIMA/weights/weights_xoftr_640.ckpt")

        elif method == "loftr":
            parser.add_argument('--ckpt', type=str,
                                default="./third/MINIMA/weights/minima_loftr.ckpt")
            parser.add_argument('--thr', type=float, default=0.2)
        elif method == "sp_lg":
            parser.add_argument('--ckpt', type=str,
                                default="./third/MINIMA/weights/minima_lightglue.pth")
        elif method == "roma":
            parser.add_argument('--ckpt2', type=str,
                                default="large")
            parser.add_argument('--ckpt', type=str, default='./third/MINIMA/weights/minima_roma.pth')
        elif method == "LoFTR_outdoor":
            parser.add_argument('--ckpt', type=str,
                                default="./third/MINIMA/weights/outdoor_ds.ckpt")
            parser.add_argument('--thr', type=float, default=0.2)
        elif method == "RoMa_outdoor":
            parser.add_argument('--ckpt2', type=str,
                                default="large")
            parser.add_argument('--ckpt', type=str, default='./third/MINIMA/weights/roma_outdoor.pth')
        elif method == "loftr_gim":
            parser.add_argument('--ckpt', type=str,
                                default="./third/gim/weights/gim_loftr_50h.ckpt")
            parser.add_argument('--thr', type=float, default=0.2)
        elif method == "RoMa_gim":
            parser.add_argument('--ckpt2', type=str,
                                default="large")
            parser.add_argument('--ckpt', type=str, default='./third/gim/weights/gim_roma_100h.ckpt')

        else:
            raise ValueError(f"Unknown method: {method}")

        add_common_arguments(parser)

    parser = argparse.ArgumentParser(description='Benchmark Relative Pose')

    parser.add_argument('--method', type=str, default='sp_lg',
                        choices=["xoftr", 'sp_lg', 'loftr', 'roma'],
                        help="Select the method to use: xoftr, sp_lg, loftr, roma")

    args, remaining_args = parser.parse_known_args()

    add_method_arguments(parser, method)

    args = parser.parse_args()
    
    if method == "LoFTR_outdoor" or method == "loftr_gim":
        matcher = load_model("loftr", args)
    elif method == "RoMa_outdoor" or method == "roma_gim":
        matcher = load_model("roma", args)
    elif method == 'xoftr640':
        matcher = load_model("xoftr", args)
    else:
        matcher = load_model(method, args)

    match_res = matcher(imgpath1, imgpath2)

    # matches = match_res['matches']
    mkpts0 = match_res['mkpts0']
    mkpts1 = match_res['mkpts1']

    return np.concatenate((mkpts0, mkpts1), axis=1), None, None