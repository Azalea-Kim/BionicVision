"""
Author: Yanxiu Jin
Date: 2025-03-17
Description: Apply thresholding to top N% of the saliency in frames and store images
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import glob
import cv2
from skimage import exposure


### Start from scene segmentation
local_dir = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master"
output_frames_dir = local_dir+"\\saliency3\\saliency_npy2image_png"
# output_frames_dir = local_dir+"\\gaze_estimations\\kitchen_20fps\\gray"

all_frames = glob.glob(output_frames_dir+"\\*.png")
# all_frames = glob.glob(output_frames_dir+"\\*.jpg")

for count in np.arange(0, len(all_frames) ):  # for deepgaze3 format
# for count in np.arange(1, len(all_frames) +1):  # each frame !!modified +1
    f_name = output_frames_dir+"\\frame_%03d_saliency.png" % count
    # f_name = output_frames_dir + "\\gray_frame_%03d.jpg" % count
    sal_img = cv2.imread(f_name, cv2.IMREAD_GRAYSCALE)
    sal = np.uint8(sal_img)


    # Threshold the saliency map
    sal_fil = sal.copy()
    threshold = np.max(sal_fil) * .60# .90
    # threshold = np.percentile(sal, 90)
    # print(threshold)
    sal_fil[sal_fil <= threshold] = 0

    # Visualize
    plt.figure(figsize=(12, 5))
    plt.imshow(sal_fil, cmap="gray")
    plt.title("Original Binary Mask (sal_fill)")
    plt.axis("off")
    plt.show()

    mask_out_dir = local_dir+"\\saliency3\\saliency_npy2image_60"

    if not os.path.exists(mask_out_dir):
        os.makedirs(mask_out_dir)
    depth_filename = os.path.join(mask_out_dir, f"frame_{count:03d}_saliency.png")
    imageio.imwrite(depth_filename, sal_fil)
