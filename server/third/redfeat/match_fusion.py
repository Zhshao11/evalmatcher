import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from third.redfeat.lib.model import MMNet
import cv2
import scipy.io as scio
from copy import deepcopy
import time
from PIL import Image
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def mutual_nn_matching_torch(desc1, desc2, threshold=None, eps=1e-9):
    if len(desc1) == 0 or len(desc2) == 0:
        return torch.empty((0, 2), dtype=torch.int64), torch.empty((0, 2), dtype=torch.int64)

    device = desc1.device
    desc1 = desc1 / (desc1.norm(dim=1, keepdim=True) + eps)
    desc2 = desc2 / (desc2.norm(dim=1, keepdim=True) + eps)
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


def MNN(desc1, desc2, threshold=None):
    if isinstance(desc1, np.ndarray):
        desc1 = torch.from_numpy(desc1)
        desc2 = torch.from_numpy(desc2)
    matches, scores = mutual_nn_matching_torch(desc1, desc2, threshold=threshold)
    return matches.cpu().numpy(), scores.cpu().numpy()


def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    model = MMNet()
    weights = checkpoint['model']
    model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return model.eval()


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.6):
        super(NonMaxSuppression,self).__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rep_thr = rep_thr
        
    def forward(self, repeatability):
        #repeatability = repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        border_mask = maxima*0
        border_mask[:,:,10:-10,10:-10]=1
        maxima = maxima*border_mask
        # print(maxima.sum())
        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, image_type,
                        scale_f=2**0.25, min_scale=0.0, 
                        max_scale=1, min_size=256, 
                        max_size=1024, border=5, verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")

            with torch.no_grad():
                if image_type == '1':
                    descriptors, repeatability = net.forward1(img)
                elif image_type == '2':
                    descriptors, repeatability = net.forward2(img)

            mask = repeatability*0
            mask[:,:,border:-border,border:-border] = 1
            repeatability=repeatability*mask
            y,x = detector(repeatability) # nms
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]
            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            #S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    #S = torch.cat(S) # scale
    scores = torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores



def mosaic_map(img1, img2, d):
    # 获取图像1的尺寸
    m1, n1, p1 = img1.shape
    m11 = int(np.ceil(m1 / d))
    n11 = int(np.ceil(n1 / d))

    # 对图像1进行处理
    for i in range(0, m11, 2):
        for j in range(1, n11, 2):
            img1[i*d:(i+1)*d, j*d:(j+1)*d, :] = 0

    for i in range(1, m11, 2):
        for j in range(0, n11, 2):
            img1[i*d:(i+1)*d, j*d:(j+1)*d, :] = 0

    # 获取处理后的图像1
    image1 = img1[0:m1, 0:n1, :]

    # 处理图像2
    m2, n2, p2 = img2.shape
    m22 = int(np.ceil(m2 / d))
    n22 = int(np.ceil(n2 / d))

    for i in range(0, m22, 2):
        for j in range(0, n22, 2):
            img2[i*d:(i+1)*d, j*d:(j+1)*d, :] = 0

    for i in range(1, m22, 2):
        for j in range(1, n22, 2):
            img2[i*d:(i+1)*d, j*d:(j+1)*d, :] = 0

    # 获取处理后的图像2
    image2 = img2[0:m2, 0:n2, :]

    # 生成拼接后的图像
    img3 = image1 + image2

    return image1, image2, img3


def image_fusion(image_1, image_2, solution):
    # Get the dimensions of the images\
    img1_np = np.array(image_1)
    img2_np = np.array(image_2)
    M1, N1, num1 = img1_np.shape
    M2, N2, num2 = img2_np.shape
    print(num1)
    print(num2)

    cv2.imwrite("img1_np0.jpg", img1_np)
    cv2.imwrite("img2_np0.jpg", img2_np)

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

    cv2.imwrite("img1_np1.jpg", img1_np)
    cv2.imwrite("img2_np1.jpg", img2_np)

    # Create an identity transformation matrix
    solution_1 = np.array([[1, 0, N1], [0, 1, M1], [0, 0, 1]],dtype=np.float32)

    # Apply the transformation to the first image
    f_1 = cv2.warpPerspective(img1_np, solution_1, (3 * N1, 3 * M1))
    cv2.imwrite("f_1.jpg", f_1)

    # Apply the transformation to the second image using the provided solution
    f_2 = cv2.warpPerspective(img2_np , solution_1 @ solution, (3 * N1, 3 * M1))
    cv2.imwrite("f_2.jpg", f_2)

    # Find overlapping regions and blend images
    same_index = np.where((f_1 != 0) & (f_2 != 0))  # 相同区域
    index_1 = np.where((f_1 != 0) & (f_2 == 0))     # 在 f_1 中而不在 f_2 中的区域
    index_2 = np.where((f_1 == 0) & (f_2 != 0))     # 在 f_2 中而不在 f_1 中的区域

    # 对相同区域进行融合，其他区域直接使用 f_1 或 f_2
    fusion_image[same_index] = f_1[same_index] // 2 + f_2[same_index] // 2
    fusion_image[index_1] = f_1[index_1]
    fusion_image[index_2] = f_2[index_2]

    # 将融合图像转换为 uint8 类型
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
        f_2= f_2[Y_min:Y_max, X_min:X_max, :]

    # save the fusion image
    cv2.imwrite("fusion.jpg", fusion_image)

    grid_num = 5  # 网格数量
    grid_size = min(f_1.shape[0], f_1.shape[1]) // grid_num  # 确定网格大小

    # 生成融合地图
    _, _, f_3 = mosaic_map(f_1, f_2, grid_size)

    # 保存融合后的图像
    cv2.imwrite('Fused_image_of_the_board.jpg', f_3)  


def redfeat(im1_path, im2_path, model_path):
    import argparse
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--num_features", type=int, default=4096, help='Number of features')
    parser.add_argument("--model", type=str, default='Pretrained/VIS_SAR.pth', help='model path')
    parser.add_argument("--img1_path", type=str, default='SAR_Optical/SO1a.png', help='path for VIS img')
    parser.add_argument("--img2_path", type=str, default='SAR_Optical/SO1b.png', help='path for other modal img')
    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--border", type=float, default=5) 
    parser.add_argument("--reliability-thr", type=float, default=0.01)
    parser.add_argument("--repeatability-thr", type=float, default=0.01)
    parser.add_argument("--gpu", type=int, default=0, help='use -1 for CPU')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    net = load_network(model_path)
    net = net.cuda()
    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr = args.reliability_thr, 
        rep_thr = args.repeatability_thr)

    img1 = Image.open(im1_path).convert('RGB')
    # img1 = cv2.imread(args.img1_path)
    W, H = img1.size
    img = TF.to_tensor(img1).unsqueeze(0)
    img = (img-img.mean(dim=[-1,-2],keepdim=True))/img.std(dim=[-1,-2],keepdim=True)
    img = img.cuda()
    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector, '1',
        scale_f   = args.scale_f, 
        min_scale = args.min_scale, 
        max_scale = args.max_scale,
        min_size  = args.min_size, 
        max_size  = args.max_size,
        border = args.border,
        verbose = False)
    if len(scores)<args.num_features:
        idxs = scores.topk(len(scores))[1]
    else:
        idxs = scores.topk(args.num_features)[1]
    kp1 = xys[idxs].cpu().numpy()
    desc1 = desc[idxs].cpu().numpy()

    img2 = Image.open(im2_path).convert('RGB')
    # img2 = cv2.imread(args.img2_path)
    W, H = img2.size
    img = TF.to_tensor(img2).unsqueeze(0)
    img = (img-img.mean(dim=[-1,-2],keepdim=True))/img.std(dim=[-1,-2],keepdim=True)
    img = img.cuda()
    
    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector, '2',
        scale_f   = args.scale_f, 
        min_scale = args.min_scale, 
        max_scale = args.max_scale,
        min_size  = args.min_size, 
        max_size  = args.max_size,
        border=args.border,
        verbose = False)
    if len(scores)<args.num_features:
        idxs = scores.topk(len(scores))[1]
    else:
        idxs = scores.topk(args.num_features)[1]
    kp2 = xys[idxs].cpu().numpy()
    desc2 = desc[idxs].cpu().numpy()

    matches_id, _ = MNN(desc1, desc2)
    src_pts = kp1[matches_id[:, 0], :2]
    dst_pts = kp2[matches_id[:, 1], :2]

    matches = np.concatenate([src_pts, dst_pts], axis=1)

    return matches

    # E, mask = cv2.findEssentialMat(
    #     src_pts, dst_pts, np.eye(3), threshold=5.0, prob=0.9999,
    #     method=cv2.RANSAC)
    # H , _= cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    # matchesMask = mask.ravel().tolist()
    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                 singlePointColor = None,
    #                 matchesMask = matchesMask, # draw only inliers
    #                 flags = 2)
    # kp1 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp1]
    # kp2 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp2]
    # img3 = cv2.drawMatches(np.array(img1),kp1,np.array(img2),kp2,good,None,**draw_params)
    # Image.fromarray(img3).save('test.png')
    # image_fusion(img1,img2,H)
    #
    # print("Essential Matrix:")
    # print(E)

