"""
Author: Yanxiu Jin
Date: 2025-03-17
Description: Combination per frame
"""


import cv2
import numpy as np
import os
import imageio

import matplotlib.pyplot  as plt

import glob

local_path = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master"
seg_path = local_path + "\\segmentation_output\\DEVA_base_clutter"
sal_path = local_path+"\\saliency3\\saliency_npy2image_png"
depth_path = local_path+"\\depth_output_npy\\kitchen20fps_TCMono_frames"


all_frames = glob.glob(seg_path+"\\*.png")

for count in np.arange(0, len(all_frames) ):
    seg_name = seg_path+"\\frame_%05d_seg.png" % count
    seg_img = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE)
    seg = np.uint8(seg_img)

    sal_name = sal_path+"\\frame_%03d_saliency.png" % count
    sal_img = cv2.imread(sal_name, cv2.IMREAD_GRAYSCALE)
    sal = np.uint8(sal_img)

    index = count + 1
    depth_name = depth_path+"\\frame_%03d_depth.png" % index
    depth_img = cv2.imread(depth_name, cv2.IMREAD_GRAYSCALE)
    dep = np.uint8(depth_img)

    # Threshold the saliency map
    sal_fil = sal.copy()
    threshold = np.max(sal_fil) * .95  # change threshold

    sal_fil[sal_fil <= threshold] = 0
    sal_fil[sal_fil > 0] = 255

    # normalize between [0,1]
    sal_norm = cv2.normalize(sal_fil, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    seg_norm = cv2.normalize(seg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dep_norm = cv2.normalize(dep, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    seg_norm = cv2.resize(seg_norm, (sal_norm.shape[1], sal_norm.shape[0]))
    dep_norm = cv2.resize(dep_norm, (sal_norm.shape[1], sal_norm.shape[0]))

    seg_sal = np.maximum(sal_norm, seg_norm)  # Element-wise max of saliency and segmentation
    dep_seg_sal = dep_norm.copy()  # No need for indexing
    dep_seg_sal[seg_sal == 0] = 0  # Apply mask


    result = dep_seg_sal.copy() * 255
    result2 = seg_sal.copy()*255
    result3 = sal_norm.copy()*255


    plt.imshow(np.uint8(result), cmap="gray")
    plt.axis("off")
    plt.title('comb')
    plt.show()

    plt.imshow(np.uint8(result2), cmap="gray")
    plt.axis("off")
    plt.title('seg_sal')
    plt.show()

    plt.imshow(np.uint8(result3), cmap="gray")
    plt.axis("off")
    plt.title('sal_norm')

    plt.show()

    mask_out_dir = local_path + "\\temporal_comb_segbase+depth+sal"
    if not os.path.exists(mask_out_dir):
        os.mkdir(mask_out_dir)

    comb_filename = os.path.join(mask_out_dir, f"frame_{count:03d}_comb.png")
    imageio.imwrite(comb_filename, np.uint8(result))
