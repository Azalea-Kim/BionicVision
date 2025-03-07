import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import glob
from skimage import exposure


### Start from scene segmentation

local_dir = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master"
output_frames_dir = local_dir+"\\saliency3\\saliency_output"
all_frames = glob.glob(output_frames_dir+"\\*.png")

for count in np.arange(0, len(all_frames) ):  # each frame !!modified +1
    f_name = output_frames_dir+"\\saliency_%05d_img.npy" % count

    depth_map = np.load(f_name)  # Shape: (H, W)

    # depth_map_eq = exposure.equalize_hist(depth_map)

    plt.imshow(depth_map, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.title('Depth Map')
    plt.show()

    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    depth_uint8 = depth_norm.astype(np.uint8)
    mask_out_dir = local_dir + "\\saliency3\\saliency_npy2image"

    if not os.path.exists(mask_out_dir):
        os.makedirs(mask_out_dir)
    depth_filename = os.path.join(mask_out_dir, f"frame_{count:03d}_saliency.jpg")
    imageio.imwrite(depth_filename, depth_uint8)
