"""
Author: Yanxiu Jin
Date: 2025-03-17
Description: Segmentation Pipeline for DEVA + saliency (priority table, persist and weighted average)
"""

import os, csv, torch, scipy.io, torchvision.transforms, glob, cv2
from collections import deque
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from skimage import morphology
# import torch
# torch.cuda.empty_cache()
from matplotlib.patches import Circle
import sys

average_scene = 10
average_objects = 5
average_arm = 5

arms_list = np.zeros((1440, 1920, average_arm))
scene_list = np.zeros((1440, 1920, average_scene))
objects_list = np.zeros((1440, 1920, average_objects))


def weighted_average(count, mask, lamb, threshold, list, average):

    print(average)
    print(list.shape)
    weights = np.array([np.exp(-lamb * i) for i in range(average - 1, -1, -1)]) #[0,1,2,3,4,5,6,7,8,9]
    print(weights)

    if count < average: # count starts from 1
        list[:, :, count - 1] = mask
        frames_to_avg = list[:, :, :count]
        weighted_avg = np.average(frames_to_avg, axis=2, weights=weights[average - count:])
    else:
        if count == average:
            list[:, :, count - 1] = mask

        else:
            list = np.roll(list, shift=-1, axis=2)
            list[:, :, -1] = mask

        print("exceed now ",count)
        frames_to_avg = list
        weighted_avg = np.average(frames_to_avg, axis=2, weights=weights)
    pixel_threshold = threshold * 255


    result_mask = np.where(weighted_avg > pixel_threshold, weighted_avg, 0)

    return result_mask, list



def PCA_get_angle(xs,ys,plot):
    points = np.column_stack((xs, ys))  # shape: (N, 2)
    mean = np.mean(points, axis=0)  # (mean_x, mean_y)
    centered = points - mean

    cov = np.cov(centered, rowvar=False)

    # PCA
    eigen_vals, eigen_vecs = np.linalg.eig(cov)
    # eigen_vals: [λ1, λ2]
    # eigen_vecs: [[v1_x, v2_x],
    #              [v1_y, v2_y]]

    idx = np.argmax(eigen_vals)
    principal_axis = eigen_vecs[:, idx]  # shape: (2,)

    # -y axis
    angle_radians = np.arctan2(principal_axis[0], principal_axis[1])
    angle_degrees = np.degrees(angle_radians)

    if plot:
        plt.figure(figsize=(6, 5))
        plt.imshow(mask_image, cmap="gray")
        plt.title(f"PCA Principal Axis (-y): {angle_degrees:.2f}°")
        plt.axis("off")
        plt.scatter(mean[0], mean[1], color='red', s=50, label='Center')

        length = 100
        x_end = mean[0] + length * principal_axis[0]
        y_end = mean[1] + length * principal_axis[1]
        plt.plot([mean[0], x_end], [mean[1], y_end], color='green', linewidth=2, label='Principal Axis')

        plt.legend()
        plt.show()

    return angle_degrees


def get_houghlines(edges):
    kernel = np.ones((10, 10), np.uint8)
    edge_history = cv2.HoughLinesP(edges.astype("uint8"), 1, np.pi / 180, 15, minLineLength=minLineLength,
                                   maxLineGap=maxLineGap)
    edge_combined = np.zeros(edges.shape)
    height, width = edges.shape
    border_threshold = 10
    min_length = 5  # define noise

    try:
        for x in range(0, len(edge_history)):
            for x1, y1, x2, y2 in edge_history[x]:
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # delete noise
                # we don't want edges in the border (seems wrong)
                if distance > min_length and not(
                        x1 < border_threshold or x2 < border_threshold or
                        y1 < border_threshold or y2 < border_threshold or
                        x1 > width - border_threshold or x2 > width - border_threshold or
                        y1 > height - border_threshold or y2 > height - border_threshold):
                            cv2.line(edge_combined, (x1, y1), (x2, y2), color=(255, 255, 255))
        edge_combined = cv2.dilate(edge_combined, kernel, iterations=1)
    except (RuntimeError, TypeError, NameError):
        print("no lines")

    return edge_combined

def exponential_circle_mask(mask_image,hand_x,hand_y,r_min,r_max,alpha, isSquaredRatio
):
    h, w = mask_image.shape[:2]

    # center of the image
    c_x, c_y = w / 2.0, h / 2.0

    # hand to center distance d
    dx = hand_x - c_x
    dy = hand_y - c_y
    d = np.sqrt(dx**2 + dy**2)

    d_max = np.sqrt((w / 2.0)**2 + (h / 2.0)**2)

    if d_max == 0:
        return r_min
    if isSquaredRatio:
        ratio = np.exp(-alpha * (d / d_max) ** 2)  # Slow Near Center, Faster at Edges
    else:
        ratio = np.exp(-alpha * (d / d_max))
    ratio = max(0, min(1, ratio))
    radius = int(r_min + (r_max - r_min) * ratio)

    return radius


def intersection_percentage(object_mask, circle_mask):
    # 1. Convert to boolean arrays for easy logical operations
    object_bool = (object_mask == 255)
    circle_bool = (circle_mask == 255)

    # 2. Count the number of 255 pixels in the object mask
    object_count = np.count_nonzero(object_bool)
    if object_count == 0:
        # No object pixels => 0% by definition
        return 0.0

    # 3. Compute intersection (where both are True)
    intersection_bool = object_bool & circle_bool
    intersection_count = np.count_nonzero(intersection_bool)

    # 4. Compute percentage
    percentage = (intersection_count / object_count) * 100.0
    return percentage


def plot_object_saliency_masks(object_bool, saliency_bool, count):
    # 1. Check shapes
    if object_bool.shape != saliency_bool.shape:
        raise ValueError("object_bool and saliency_bool must have the same shape.")

    H, W = object_bool.shape

    # 2. Create an RGB image (H,W,3) filled with 0 (black)
    color_image = np.zeros((H, W, 3), dtype=np.uint8)

    # 3. Intersection => Yellow
    intersection = object_bool & saliency_bool
    color_image[intersection] = (255, 255, 0)  # BGR= (0,255,255) if using cv2, but here it's RGB

    if not intersection.any():
        return
        # 4. Object only => Green
    object_only = object_bool & ~intersection
    color_image[object_only] = (0, 255, 0)

    # 5. Saliency only => Red
    saliency_only = saliency_bool & ~intersection
    color_image[saliency_only] = (255, 0, 0)

    # 6. Plot with Matplotlib
    plt.figure(figsize=(6, 5))
    plt.imshow(color_image)
    plt.title(str(count))
    plt.axis("off")

    plt.show()
    print(count)
    # mask_out_dir = "segmentation_output/deva_sal_intersect_70"
    # if not os.path.exists(mask_out_dir):
    #     os.mkdir(mask_out_dir)
    # masks_comb_uint8 = color_image.astype(np.uint8)
    # seg_filename = os.path.join(mask_out_dir, f"frame_{count:05d}_seg.png")
    # imageio.imwrite(seg_filename, masks_comb_uint8)

def intersection_saliency(object_mask, saliency_mask, count):
    # 1. Convert to boolean arrays for easy logical operations
    object_bool = (object_mask >0 )
    saliency_bool = (saliency_mask >  0)
    # plot_object_saliency_masks(object_bool, saliency_bool, count)
    # 2. Count the number of 255 pixels in the object mask
    object_count = np.count_nonzero(object_bool)
    if object_count == 0:
        # No object pixels => 0% by definition
        return 0.0

    # 3. Compute intersection (where both are True)
    intersection_bool = object_bool & saliency_bool
    intersection_count = np.count_nonzero(intersection_bool)

    return intersection_count > 0


def plot_segmentation_classes(seg_array):
    # 2. Find unique class IDs
    unique_classes = np.unique(seg_array)
    print(f"Total classes found: {len(unique_classes)}")
    print(f"Class IDs: {unique_classes}")

    # 3. Plot each class mask
    num_classes = len(unique_classes)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))

    # If there's only 1 class, axes might not be iterable
    if num_classes == 1:
        axes = [axes]  # make it a list for consistency

    for i, class_id in enumerate(unique_classes):
        # Create a binary mask where seg_array == class_id
        class_mask = (seg_array == class_id).astype(np.uint8)  # 1 for that class, 0 elsewhere
        # Plot
        axes[i].imshow(class_mask, cmap='gray')
        axes[i].set_title(f"Class {class_id}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def get_nonzero_class_masks(seg_array):
    # 1. Find all unique classes

    unique_classes = np.unique(seg_array)

    # 2. Filter out the 0 (background)
    nonzero_classes = unique_classes[unique_classes != 0]

    # 3. Build a list of masks
    masks = []
    for cls_id in nonzero_classes:
        mask = (seg_array == cls_id).astype(np.uint8)  # 1 where seg_array==cls_id, else 0
        masks.append(mask)

    return masks


local_dir = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master"
arm_frames_dir = local_dir+"\\deva_outputs\\masks\\masks\\arm"
scene_frames_dir = local_dir+"\\deva_outputs\\masks\\masks\\door.window"
objects_frames_dir = local_dir+"\\deva_outputs\\masks\\masks\\utensil.food.kitchen_appliance.pot.pan.knife.cutting_board"
all_frames = glob.glob(arm_frames_dir+"\\*.npy")
saliency_frames_dir = local_dir+"\\saliency3\\saliency_npy2image_95"
# Deal with cases with error in hand detection
person_history = deque([False, False, False, False, False], maxlen=5)
most_recent_circle_mask = None

saliency_history = deque([False, False, False, False, False, False, False, False, False, False], maxlen=10)
most_recent_saliency_mask = None


# Get bright if no hand is detected for long
no_hand_frames = 0
no_hand_threshold = 30

# persist
average_scene = 10
average_objects = 5
average_arm = 5

arms_list = np.zeros((1440, 1920, average_arm))
scene_list = np.zeros((1440, 1920, average_scene))
objects_list = np.zeros((1440, 1920, average_objects))



for count in np.arange(0, len(all_frames)):  # each frame !!modified +1
# for count in np.arange(0, 13):  # each frame !!modified +1
    arm_name = arm_frames_dir+"\\frame_%05d.npy" % count
    arm_mask = np.load(arm_name)

    scene_name = scene_frames_dir + "\\frame_%05d.npy" % count
    scene_mask = np.load(scene_name)

    objects_name = objects_frames_dir + "\\frame_%05d.npy" % count
    objects_mask = np.load(objects_name)

    saliency_name = saliency_frames_dir + "\\frame_%03d_saliency.png" % count
    # saliency_name = saliency_frames_dir + "\\frame_%03d_saliency.png" % (count+1)
    sal_img = cv2.imread(saliency_name, cv2.IMREAD_GRAYSCALE)
    saliency_mask = np.uint8(sal_img)

    # plot_segmentation_classes(arm_mask)
    arm = get_nonzero_class_masks(arm_mask)
    objects = get_nonzero_class_masks(objects_mask)
    scene = get_nonzero_class_masks(scene_mask)


    # Persist
    num_instances = max(num_instances, np.max(objects_mask))
        print(num_instances)
        images.append(objects_mask)
        num_instances = int(num_instances)
        if count ==0 :
            instance_seen_last = [-1]*num_instances

        else:
            if len(instance_seen_last) < num_instances:
                instance_seen_last.extend([-1] * (num_instances - len(instance_seen_last)))

        print(instance_seen_last)

        index = count
        print(index)
        persist = 10
        for i in range(num_instances):
            instance_id = i + 1
            if instance_id in objects_mask:
                instance_seen_last[i] = index
            elif instance_seen_last[i] != -1 and index - instance_seen_last[i] <= persist:
                prev_frame = images[instance_seen_last[i]]
                mask = (prev_frame == instance_id)

                curr_vals = images[index][mask].copy()

                images[index][mask] = np.maximum(curr_vals, instance_id)

                objects_mask = images[index]

    # plot_segmentation_classes(scene_mask)
    # plot_segmentation_classes(objects_mask)

    # Final combination mask
    masks_comb = np.zeros_like(arm_mask)
    min_area = 80000
    is_person = False

    # generate hand mask
    if len(arm) > 0:
        is_person = True
        hand_mask = np.zeros_like(arm[0])
        circle_mask = np.zeros_like(arm[0])

        for i in arm:
            mask_image = i * 255
            hand_mask = np.maximum(hand_mask, i * 255)

            ys, xs = np.where(mask_image == 255)
            # Safety check in case the mask is empty
            if len(xs) == 0 or len(ys) == 0:
                raise ValueError("Mask appears empty or no 255 region found!")

            y_min, y_max = np.min(ys), np.max(ys)
            x_min, x_max = np.min(xs), np.max(xs)

            width = x_max - x_min
            height = y_max - y_min

            radius = np.sqrt(width ** 2 + height ** 2) / 2.0

            angle_degrees = PCA_get_angle(xs, ys, False)

            if angle_degrees < 0:
                centroid_x, centroid_y = x_min, y_min
            elif angle_degrees > 0:
                centroid_x, centroid_y = x_max, y_min
            else:
                centroid_x, centroid_y = (x_min + x_max) / 2, y_min

            radius = exponential_circle_mask(mask_image, centroid_x, centroid_y, 100, 400, 2.0, True)

            c_mask = np.zeros_like(circle_mask).astype(np.uint8)

            # Draw an outline circle (not filled) with white=255
            cv2.circle(c_mask, (int(centroid_x), int(centroid_y)), int(radius), 255, thickness=-1)

            circle_mask = np.maximum(circle_mask, c_mask)
            most_recent_circle_mask = circle_mask

    # deal with object instances
    if len(objects) > 0:
        for mask in objects:
                s_weight = 1.00
                is_saliency = False
                if intersection_saliency(mask, saliency_mask, count) == True:
                    is_saliency = True
                    s_weight = 1.159

                # has saliency intersection in past 10 frames
                if is_saliency == False and any(saliency_history):
                    is_saliency = intersection_saliency(mask, most_recent_saliency_mask, count)
                    if is_saliency == True:
                        s_weight = 1.159
                        print("Recent Saliency")

                area = np.sum(mask)
                ip = 0.0
                b = 160  # Base:b2
                if is_person:
                    # hand detected
                    ip = intersection_percentage(mask * 255, circle_mask)
                elif any(person_history):
                    # hand detected in past fixed frames
                    ip = intersection_percentage(mask * 255, most_recent_circle_mask)
                # No hand detected for long time get back to 220 not 160
                elif no_hand_frames >= no_hand_threshold:
                    b = 220

                if ip > 0.50:
                    b = 220  # b12 only hand circle intersect

                if area > min_area:
                    # edge detection
                    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_edge = mask_image.filter(ImageFilter.FIND_EDGES)
                    mask_edge = np.array(mask_edge)

                    kernel = np.ones((20, 20), np.uint8)  # unit8 [0,255]
                    mask_edge = cv2.dilate(mask_edge, kernel, iterations=1)  # make edges thicker and continuous
                    # add edges of big objects

                    mask_edge = mask_edge.astype(np.float32)  # convert to float for scaling
                    mask_edge *= (float(b)*s_weight) / 255.0  # scale to [0..200]
                    mask_edge = mask_edge.astype(np.uint8)  # convert back to uint8

                    masks_comb = np.maximum(masks_comb, mask_edge)
                else:
                    # add not hand region mask

                    # if it's not big objects, extract edges and lower intensity for contrast
                    if is_saliency:
                        mask_i = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_e = mask_i.filter(ImageFilter.FIND_EDGES)
                        mask_e = np.array(mask_e)
                        kernel = np.ones((7, 7), np.uint8)  # unit8 [0,255]
                        mask_e = cv2.dilate(mask_e, kernel, iterations=1)  # make edges thicker and continuous

                        edge_val = (b / s_weight)

                        edge_array = np.where(mask_e == 255, edge_val, 0)

                        # temp = mask.astype(np.float32) * (b * s_weight)
                        temp = mask.astype(np.float32) * 255
                        masks_comb = np.maximum(masks_comb.astype(np.float32), temp)
                        masks_comb = np.clip(masks_comb, 0, 255).astype(np.uint8)

                        #lower the intensity on edge to enhance contrast
                        # Override where edge_array is > 0
                        override_mask = (edge_array > 0)
                        masks_comb[override_mask] = edge_array[override_mask]
                        masks_comb = np.clip(masks_comb, 0, 255).astype(np.uint8)

                        most_recent_saliency_mask = saliency_mask
                        saliency_history.append(True)
                    else:
                        masks_comb = np.maximum(masks_comb, mask * b)
                        saliency_history.append(False)


    if is_person:
            # weighted average
            masks_comb, objects_list = weighted_average(count + 1, masks_comb, 1.2, 0.1, objects_list, average_objects)
            hand_mask, arms_list = weighted_average(count + 1, hand_mask, 0.8, 0.3, arms_list, average_arm)
            masks_comb = np.maximum(masks_comb, hand_mask)

    plt.figure(figsize=(12, 5))
    plt.imshow(masks_comb, cmap="gray")
    plt.title("Original Binary Mask (sal_fill)")
    plt.axis("off")
    plt.show()


    if len(scene) > 0:
        # filter out small islands
        pred_clean2 = morphology.remove_small_objects(scene_mask.astype(bool), min_size=16000).astype(int) * 255
        pred_clean3 = np.minimum(scene_mask, pred_clean2)

        # get structure edges and get only long ones
        image = Image.fromarray(np.uint8((pred_clean3+1) * 255), 'L') # turn to black and white
        image_edge = image.filter(ImageFilter.FIND_EDGES) # edge detection
        image_edge = np.array(image_edge)
        kernel = np.ones((10, 10), np.uint8) # unit8 [0,255]
        image_edge = cv2.dilate(image_edge, kernel, iterations=1) # make edges thicker and continuous

        edges_uint8 = image_edge.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_uint8, connectivity=8)
        area_threshold = 1500
        edges_filtered = np.zeros_like(edges_uint8)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= area_threshold:
                edges_filtered[labels == label] = 255

        image_edge = edges_filtered

        minLineLength = 5
        maxLineGap = 1
        lines = cv2.HoughLinesP(image_edge, 1, np.pi / 180, 15, minLineLength=minLineLength, maxLineGap=maxLineGap)
        edges = np.zeros(pred_clean3.shape) # initialize a blank image
        try: #modified here
            height, width = edges.shape
            border_threshold = 10
            min_length = 30  # define noise

            for x in range(0, len(lines)): # a line consists of two points (x1, y1) (x2, y2)
                for x1, y1, x2, y2 in lines[x]:
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)   # delete noise

                    # we don't want edges in the border (seems wrong)
                    if distance > min_length and not(
                            x1 < border_threshold or x2 < border_threshold or
                            y1 < border_threshold or y2 < border_threshold or
                            x1 > width - border_threshold or x2 > width - border_threshold or
                            y1 > height - border_threshold or y2 > height - border_threshold):
                        cv2.line(edges, (x1, y1), (x2, y2), color=(255, 255, 255))
            edges = cv2.dilate(edges, kernel, iterations=1)


        except (RuntimeError, TypeError, NameError):
            print("no lines")

        kernel = np.ones((10, 10), np.uint8)
        # erode to reduce noise
        edges = cv2.erode(get_houghlines(edges), kernel)

        edges_uint8 = edges.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_uint8, connectivity=8)
        area_threshold = 1500
        edges_filtered = np.zeros_like(edges_uint8)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= area_threshold:
                edges_filtered[labels == label] = 255

        edges = edges_filtered

        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # morphologyEx can reduce smaller noise, make edges smoother
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel2, iterations=3)

        # hough and erode strengthen edges structures and reduce noise
        edges = cv2.erode(get_houghlines(edges), kernel)

        ### add scene edges to the frame
        if is_person:
            b = 160 # b2

        else:
            b = 255

        scene_edges = edges.astype(np.float32)  # convert to float for scaling
        scene_edges *= float(b) / 255.0  # scale to [0..200]
        scene_edges = scene_edges.astype(np.uint8)  # convert back to uint8

        scene_edges, scene_list = weighted_average(count+1, scene_edges, 1.0, 0.3, scene_list, average_scene)

        masks_comb = np.maximum(masks_comb, scene_edges)

    if is_person:
        person_history.append(True)
        no_hand_frames = 0
    else:
        person_history.append(False)
        no_hand_frames += 1


    # Create output folder if it doesn't exist
    mask_out_dir = "segmentation_output/DEVA_base_clutter"
    if not os.path.exists(mask_out_dir):
        os.mkdir(mask_out_dir)

    # Display and save segmented image
    print("Processing frame %d" % count)
    plt.imshow(masks_comb, cmap="gray",vmin=0, vmax=255)
    plt.axis("off")
    plt.title('Object segmentation (b2=160, b12=220, ip>50)')

    plt.show()
    # filename = "frame_%d_seg.png" % count
    # filepath = os.path.join(mask_out_dir, filename)
    # plt.savefig(filepath, bbox_inches='tight', pad_inches=0)

    masks_comb_uint8 = masks_comb.astype(np.uint8)
    seg_filename = os.path.join(mask_out_dir, f"frame_{count:05d}_seg.png")
    imageio.imwrite(seg_filename, masks_comb_uint8)


