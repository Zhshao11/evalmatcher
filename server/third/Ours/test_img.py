from time import time
from PIL import Image
import numpy as np
import imageio as imio
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt

import viz2d
from third.Ours.lib.xfeat import XFeat


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]
    print('Inliner Matches: ', len(matches))

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches


xfeat = XFeat(weights='trained_ckpt/model/2024_09_25-15_39_34/VIS_SAR_88000.pth')

#Load some example images
im1 = cv2.imread('VIS_SAR/test/VIS/110.png')
im2 = cv2.imread('VIS_SAR/test/SAR/110.png')

# im1 = Image.open('VIS_SAR/test/VIS/110.png').convert('RGB')
# im2 = Image.open('VIS_SAR/test/SAR/110.png').convert('RGB')
# # im2 = im2.rotate(60)
# im2 = im2.resize((600, 600))
# #
# im1 = np.asarray(im1)
# im2 = np.asarray(im2)

#Use out-of-the-box function for extraction + MNN matching
tic = time()
matches, kp1, kp2 = xfeat.match_cmodel(im1, im2, top_k=4096)
mkpts_0, mkpts_1 = matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy()

canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
toc = time()
print('Running Time: ', toc - tic)
cv2.imwrite('test_result.jpg', canvas)

# viz_keypoint
kp1 = kp1.squeeze(0).cpu().numpy()
kp2 = kp2.squeeze(0).cpu().numpy()

viz2d.plot_images([np.asarray(im1)])
viz2d.plot_keypoints([kp1])
viz_path = 'keypoints1.png'
viz2d.save_plot(viz_path)
print('keypoints1 num: ', len(kp1))

viz2d.plot_images([np.asarray(im2)])
viz2d.plot_keypoints([kp2])
viz_path = 'keypoints2.png'
viz2d.save_plot(viz_path)
print('keypoints2 num: ', len(kp2))

# plt.figure(figsize=(12,12))
# plt.imshow(canvas[..., ::-1]), plt.show()