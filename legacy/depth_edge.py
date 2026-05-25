"""
Author: Yanxiu Jin
Date: 2025-03-17
Description: Edge detectoin from depth map based on
the paper 'LBP-Based Edge Detection Method for Depth Images With Low Resolutions'
Didn't apply for final result
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import imageio
import os
from skimage.morphology import skeletonize
from skimage import morphology

def apply_contrast_stretching(image, low, high):
    # Ensure the low and high values are within the valid range [0, 255]
    # Create a copy of the image to avoid modifying the original
    stretched_image = image.copy()

    # Apply the contrast stretching to each pixel
    stretched_image = np.where(stretched_image < low, 0, stretched_image)
    stretched_image = np.where((low <= stretched_image) & (stretched_image <= high),
                              (255 / (high - low)) * (stretched_image - low), stretched_image)
    stretched_image = np.where(stretched_image > high, 255, stretched_image)

    return stretched_image

def calculate_L(j, i_T):
    return 1 if j >= i_T else 0


def calculate_D(k):
    return 1 if k >= 1 else 0


def calculate_DLBP(center_pixel, neighbors):
    i_max = np.max(neighbors)
    i_aver = np.mean(neighbors)
    i_T = i_max - i_aver
    LBP_value = sum(calculate_L(neighbor - center_pixel, i_T) for neighbor in neighbors)
    DLBP_value = calculate_D(LBP_value)
    return DLBP_value


def calculate_R(center_pixel, neighbors, threshold=1.5):
    i_aver = np.mean(neighbors)
    return calculate_DLBP(center_pixel, neighbors) if abs(center_pixel - i_aver) >= threshold else 0

local_path = 'D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master'
image = cv2.imread(local_path+'\\depth_output_npy\\kitchen20fps_TCMono_frames\\frame_001_depth.png', cv2.IMREAD_GRAYSCALE)
# Check if the image was loaded successfully

depth_path = local_path+"\\depth_output_npy\\kitchen20fps_TCMono_frames"


all_frames = glob.glob(depth_path+"\\*.png")
# all_frames = glob.glob(output_frames_dir+"\\*.jpg")

for count in np.arange(1, len(all_frames) +1):  # each frame !!modified +1
    depth_name = depth_path+"\\frame_%03d_depth.jpg" % count
    depth_img = cv2.imread(depth_name, cv2.IMREAD_GRAYSCALE)
    image = np.uint8(depth_img)

    # Create an output image to store the R values
    output_image = np.zeros_like(image)

    # Define the neighborhood size (3x3)
    neighborhood_size = 3 #3

    # Iterate through the image pixels, applying DLBP and edge detection
    for y in range(neighborhood_size, image.shape[0] - neighborhood_size):
        for x in range(neighborhood_size, image.shape[1] - neighborhood_size):
            center_pixel = image[y, x]
            neighbors = [image[y - 1, x - 1], image[y - 1, x], image[y - 1, x + 1],
                         image[y, x - 1], image[y, x + 1],
                         image[y + 1, x - 1], image[y + 1, x], image[y + 1, x + 1]]

            R_value = calculate_R(center_pixel, neighbors, 1.2)
            output_image[y, x] = R_value

    # Save or display the resulting edge map (output_image)
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(output_image, cmap='gray')
    plt.title('DBLP')
    # Stretched Image and Histogram

    plt.show()

    plt.imshow(output_image, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.title('output')
    plt.show()

    # Example: A horizontal line kernel of length 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

    closed_horiz = cv2.morphologyEx(output_image, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(closed_horiz, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('closed')
    # plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    closed_vert = cv2.morphologyEx(closed_horiz, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(closed_vert, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('closed_all')
    # plt.show()



    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thick = cv2.dilate(closed_vert, kernel, iterations=1)

    # plt.imshow(thick, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('thick1')
    # plt.show()


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))

    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(thick, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('closed')
    # plt.show()


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))

    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(thick, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('closed')
    # plt.show()


    kernel = np.ones((3,3), np.uint8)

    thick  = cv2.dilate(thick, kernel, iterations=1)

    # plt.imshow(thick, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('dilate')
    # plt.show()

    thick   = cv2.erode(thick, kernel, iterations=1)

    # plt.imshow(thick, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('erode')
    # plt.show()

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    thick = cv2.morphologyEx(thick, cv2.MORPH_OPEN, kernel)
    # plt.imshow(thick, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('opened3')
    # plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    closed_horiz = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(closed_horiz, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('closed2')
    # plt.show()

    bool_img = (closed_horiz > 0)
    #    This step discards connected components smaller than 'min_size' pixels.
    cleaned_bool = morphology.remove_small_objects(bool_img, min_size=200)

    # 3) Convert back to uint8 (0/255)
    closed_horiz = (cleaned_bool * 255).astype(np.uint8)

    # plt.imshow(closed_horiz, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('cleaned1')
    # plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
    thick = cv2.morphologyEx(closed_horiz, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(thick, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('closed_all2')
    # plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilated = cv2.dilate(thick, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    min_area = 100
    filtered = np.zeros_like(labels, dtype=np.uint8)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == lbl] = 255

    # plt.imshow(filtered, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('filtered')
    # plt.show()

    bool_img = (filtered > 0)
    cleaned_bool = morphology.remove_small_objects(bool_img, min_size=2000)
    cleaned_uint8 = (cleaned_bool * 255).astype(np.uint8)

    plt.imshow(cleaned_uint8, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.title('cleaned')
    plt.show()

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(cleaned_uint8, cv2.MORPH_CLOSE, kernel)


    plt.imshow(closed, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.title('final')
    plt.show()

    # Apply skeletonization to get the thinnest possible structure
    skeleton = skeletonize(closed)

    # Convert skeleton back to uint8 format for visualization
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    # plt.imshow(skeleton_uint8, cmap='gray')
    # plt.axis('off')  # Hide axis
    # plt.title('skeleton_uint8')
    # plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    # Apply dilation to thicken the skeletonized edges slightly
    # closed = cv2.dilate(skeleton_uint8, np.ones((3,3), np.uint8), iterations=5)
    closed = cv2.dilate(skeleton_uint8,kernel, iterations=5)

    plt.imshow(closed, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.title('skeleton')
    plt.show()

    thick = cv2.erode(thick, kernel, iterations=1)

    mask_out_dir = local_path + "\\depth_edges"
    if not os.path.exists(mask_out_dir):
        os.mkdir(mask_out_dir)

    comb_filename = os.path.join(mask_out_dir, f"frame_{count:03d}_edge.png")
    # comb_filename = os.path.join(mask_out_dir, f"frame_011_comb_temp.png")
    imageio.imwrite(comb_filename, np.uint8(closed))




