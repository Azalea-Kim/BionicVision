import math
import numpy as np
import cv2

# remap x to a certain range
def restrict_x_value(d, max_d, min_d, target_max_d, target_min_d):
    k = (target_min_d - target_max_d) / (min_d - max_d)
    b = target_min_d - k * min_d
    return k*d+b

# naive linear map; remap x to [2, 10] and then y = kx + b fit to (2, max_y) and (10, min_y)
def depth_linear_map(d, max_depth, min_depth, min_y, max_y):
    d = restrict_x_value(d, max_depth, min_depth, 10, 2)
    k = (min_y - max_y) / (10 - 2)
    b = min_y - k * 10
    return k*d+b

# quadratic map; remap x to [2, 10] and then y = a(x-h)^2 + k fit to (2, max_y) and (10, min_y) as the vertex
def depth_quadratic_map(depth_map, max_depth, min_depth, min_y, max_y):
    depth_map = restrict_x_value(depth_map, max_depth, min_depth, 10, 2)
    a = (max_y - min_y) / math.pow(2 - 10, 2)
    return a * np.power(depth_map - 10, 2) + min_y

# flipped quadratic map; remap x to [2, 10] and then y = a(x-h)^2 + k fit to (10, min_y) and (2, max_y) as the vertex
def depth_flipped_quadratic_map(depth_map, max_depth, min_depth, min_y, max_y):
    depth_map = restrict_x_value(depth_map, max_depth, min_depth, 10, 2)
    a = (min_y - max_y) / math.pow(10 - 2, 2)
    return a * np.power(depth_map - 2, 2) + max_y

# exponential map; remap x to [2, 10] and y = a*e^(-x) + b fit to (2, max_y) and (10, min_y)
def depth_exponential_map(depth_map, max_depth, min_depth, min_y, max_y):
    depth_map = restrict_x_value(depth_map, max_depth, min_depth, 10, 2)
    a = (max_y - min_y) / (math.exp(-2) - math.exp(-10))
    b = max_y - a * math.exp(-2)

    return a * np.exp(-depth_map) + b

# flipped exponential map; remap x to [2, 10] and y = a*e^x + b fit to (2, max_y) and (10, min_y)
def depth_flipped_exponential_map(depth_map, max_depth, min_depth, min_y, max_y):
    depth_map = restrict_x_value(depth_map, max_depth, min_depth, 10, 2)
    a = (max_y - min_y) / (math.exp(2) - math.exp(10))
    b = max_y - a * math.exp(2)
    return a * np.exp(depth_map) + b

# generate images with depths encoded by brightness
# sidewalk_mask is used to indicated the region to retain; all the pixles outside the mask will be set to 0.
# In the current usage, sidewalk_mask is set to be all 1's which means everything is retained.
def gen_image_brightness(depth_map, sidewalk_mask, mode="linear", min_brightness=50, max_brightness=255):
    if np.count_nonzero(sidewalk_mask) == 0:
        return np.zeros((sidewalk_mask.shape[0], sidewalk_mask.shape[1], 3)).astype(np.uint8)
    max_depth = depth_map[sidewalk_mask].max()
    min_depth = depth_map[sidewalk_mask].min()
    if mode == "quadratic":
        one_channel = depth_quadratic_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    elif mode == "exponential":
        one_channel = depth_exponential_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    elif mode == "flipped_quadratic":
        one_channel = depth_flipped_quadratic_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    elif mode == "flipped_exponential":
        one_channel = depth_flipped_exponential_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    else:
        one_channel = depth_linear_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    complement_sidewalk_mask = sidewalk_mask == False
    one_channel[complement_sidewalk_mask] = 0
    im = np.array([one_channel] * 3).astype(np.uint8)

    # print("Original depth shape:", depth_map.shape)
    # print("Processed one_channel shape:", one_channel.shape)
    return np.transpose(im, (1, 2, 0))

import os
import numpy as np
import imageio


def gen_depth_videos(folder, mode, min_brightness, max_brightness, clipped=False, p=95, save_frames_folder=""):
    # # num_frames = len(os.listdir(folder))
    # num_frames = len([f for f in os.listdir(folder) if f.endswith("_disp.jpeg")])
    # # depth_list = [np.squeeze(np.load("{}\\frame_{:03d}_disp.npy".format(folder, i))) for i in range(1,num_frames+1)]
    # ## !!! problem still too bright
    # depth_list = [np.squeeze(np.load("{}\\frame_{:03d}_depth.npy".format(folder, i))) for i in range(1, num_frames + 1)]
    # # disparity value big in near place
    # depth_shape = depth_list[0].shape

    # For temporal depth
    # num_frames = len(os.listdir(folder))
    num_frames = len([f for f in os.listdir(folder) if f.endswith(".jpeg")])
    print(num_frames)
    # depth_list = [np.squeeze(np.load("{}\\frame_{:03d}_disp.npy".format(folder, i))) for i in range(1,num_frames+1)]
    ## !!! problem still too bright
    ## change file names if needed
    depth_list = [np.squeeze(np.load("{}\\frame_{:03d}_depth.npy".format(folder, i))) for i in range(1, num_frames + 1)]
    # disparity value big in near place
    depth_shape = depth_list[0].shape


    if clipped:
        # make all the values above p percentile to be the value of p percentile
        depth_clipped = []
        for depth in depth_list:
            vmax = np.percentile(depth, p)
            depth[depth > vmax] = vmax
            depth_clipped.append(depth)
        final_images = [gen_image_brightness(depth_clipped[i], np.ones(depth_shape, dtype=bool), mode=mode, min_brightness=min_brightness, max_brightness=max_brightness) for i in range(num_frames)]
    else:
        final_images = [gen_image_brightness(depth_list[i], np.ones(depth_shape, dtype=bool), mode=mode, min_brightness=min_brightness, max_brightness=max_brightness) for i in range(num_frames)]

    # save frames
    if save_frames_folder:
        if not os.path.exists(save_frames_folder):
            os.makedirs(save_frames_folder)
        for i, img in enumerate(final_images):
            frame_filename = os.path.join(save_frames_folder, f"frame_{i+1:03d}.jpeg")
            imageio.imwrite(frame_filename, img)

    w = imageio.get_writer(save_frames_folder+"kitchen20fps_monodepth2.mp4", mode='I', fps=20)
    for i in range(len(final_images)):
        w.append_data(final_images[i])
    w.close()

folder = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\depth_output_npy\\kitchen20fps_monodepth2"
save_frames_folder ="D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\depth_output_npy\\kitchen20fps_monodepth2_frames\\"
# gen_depth_videos(folder, mode="linear", min_brightness=0, max_brightness=180, clipped=True, p=80)
# gen_depth_videos(folder, mode="quadratic", min_brightness=0, max_brightness=180, clipped=False, p=80, save_frames_folder=save_frames_folder)
gen_depth_videos(folder, mode="exponential", min_brightness=0, max_brightness=180, clipped=False, p=80, save_frames_folder=save_frames_folder)

# Exponential mapping makes depth changes more dramatic at close distances while becoming more gradual at farther distances, aligning with human perception.
# It is the most natural way to represent depth perception.

# Disparity maps exhibit dramatic changes in close-range areas while remaining nearly unchanged in distant regions,
# making them suitable for short-distance scenarios such as indoor environments.


# use quadratic when disp





