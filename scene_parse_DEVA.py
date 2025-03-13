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



def PCA_get_angle(xs,ys,plot):
    # 2. 组合成点集, 形状 (N, 2)
    # 注意：这里把 (x, y) 放在一起，符合常见的 (col, row) => (x, y) 约定
    points = np.column_stack((xs, ys))  # shape: (N, 2)

    # 3. 计算均值, 并将点集中心化
    mean = np.mean(points, axis=0)  # (mean_x, mean_y)
    centered = points - mean

    # 4. 计算 2×2 协方差矩阵
    cov = np.cov(centered, rowvar=False)  # rowvar=False 表示每列是一个维度

    # 5. 对协方差矩阵做特征分解 (PCA)
    eigen_vals, eigen_vecs = np.linalg.eig(cov)
    # eigen_vals: [λ1, λ2]
    # eigen_vecs: [[v1_x, v2_x],
    #              [v1_y, v2_y]]

    # 取最大特征值对应的特征向量 (主轴)
    idx = np.argmax(eigen_vals)
    principal_axis = eigen_vecs[:, idx]  # shape: (2,)

    # 6. 计算主轴的旋转角度 (相对于 x 轴)
    angle_radians = np.arctan2(principal_axis[0], principal_axis[1])
    angle_degrees = np.degrees(angle_radians)
    # print(f"主轴角度: {angle_degrees:.2f}° (相对于 -y 轴)")

    if plot:
        # 7. 可视化结果
        plt.figure(figsize=(6, 5))
        plt.imshow(mask_image, cmap="gray")
        plt.title(f"PCA Principal Axis (-y): {angle_degrees:.2f}°")
        plt.axis("off")

        # 绘制中心点
        plt.scatter(mean[0], mean[1], color='red', s=50, label='Center')

        # 从中心点延伸出一条表示主轴方向的线
        length = 100  # 线段长度，可根据图像尺寸调节
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

def exponential_circle_mask(
    mask_image,
    hand_x,
    hand_y,
    r_min,
    r_max,
    alpha, isSquaredRatio
):
    """
    alpha : float
        控制指数衰减的系数。越大表示半径随距离衰减更快。

    """
    # 1. 获取图像宽高
    h, w = mask_image.shape[:2]

    # 2. 计算图像中心
    c_x, c_y = w / 2.0, h / 2.0

    # 3. 计算手到中心的距离 d
    dx = hand_x - c_x
    dy = hand_y - c_y
    d = np.sqrt(dx**2 + dy**2)

    # 4. 最大距离 d_max (从中心到四角中最远的一个角)
    d_max = np.sqrt((w / 2.0)**2 + (h / 2.0)**2)

    # 5. 指数衰减计算 ratio
    # ratio 在 d=0 时 = 1.0, d= d_max 时 ~= e^(-alpha)
    if d_max == 0:
        # 容错：万一图像尺寸为 0
        return r_min

    # 让 ratio = exp(-alpha * (d / d_max))
    if isSquaredRatio:
        ratio = np.exp(-alpha * (d / d_max) ** 2)  # Slow Near Center, Faster at Edges
    else:
        ratio = np.exp(-alpha * (d / d_max))


    # 6. 限制 ratio 在 [0,1]
    ratio = max(0, min(1, ratio))

    # 7. 根据 ratio 计算最终半径
    radius = int(r_min + (r_max - r_min) * ratio)

    return radius



def intersection_percentage(object_mask: np.ndarray, circle_mask: np.ndarray) -> float:
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
    """
    object_bool: 2D boolean array (True/False)
    saliency_bool: 2D boolean array (True/False)

    We create a 3-channel color image (H,W,3) in uint8:
      - Intersection (object & saliency) => Yellow (255,255,0)
      - Object only => Green (0,255,0)
      - Saliency only => Red (255,0,0)
      - Neither => Black (0,0,0)
    Then we plot it with Matplotlib.
    """
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
    plot_object_saliency_masks(object_bool, saliency_bool, count)
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

# saliency_frames_dir = local_dir+"\\gaze_estimations\\kitchen_20fps\\threshold_90" # 1
#
#



# Deal with cases with error in hand detection
person_history = deque([False, False, False, False, False], maxlen=5)
most_recent_circle_mask = None

saliency_history = deque([False, False, False, False, False, False, False, False, False, False], maxlen=10)
most_recent_saliency_mask = None

# Get bright if no hand is detected for long
no_hand_frames = 0
no_hand_threshold = 30

W = 10  # store the most recent W frames' edge detection result
w_count = 0
edge_rep = np.zeros((1440, 1920, W))


for count in np.arange(189, len(all_frames)):  # each frame !!modified +1
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

    # plot_segmentation_classes(scene_mask)
    # plot_segmentation_classes(objects_mask)

    # Final combination mask
    masks_comb = np.zeros_like(arm_mask)
    min_area = 80000
    is_person = False

    if len(arm) > 0:
        is_person = True
        hand_mask = np.zeros_like(arm[0])
        circle_mask = np.zeros_like(arm[0])

        for i in arm:
            mask_image = i * 255
            hand_mask = np.maximum(hand_mask, i * 255)  # 单独分出来后面可以少加一次

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
                    # print("有手")
                    ip = intersection_percentage(mask * 255, circle_mask)
                elif any(person_history):
                    # print("前两frame有手")
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

                        # edge_val = (b / 2)
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

        masks_comb = np.maximum(masks_comb, scene_edges)

    if is_person:
        person_history.append(True)
        no_hand_frames = 0
    else:
        person_history.append(False)
        no_hand_frames += 1


    # Create output folder if it doesn't exist
    mask_out_dir = "segmentation_output/deva_seg_scene_sal_95_255_history"
    if not os.path.exists(mask_out_dir):
        os.mkdir(mask_out_dir)

    # Display and save segmented image
    print("Processing frame %d" % count)
    plt.imshow(masks_comb, cmap="gray",vmin=0, vmax=255)
    plt.axis("off")
    plt.title('Object segmentation (b2=160, b12=220, ip>50)')


    # filename = "frame_%d_seg.png" % count
    # filepath = os.path.join(mask_out_dir, filename)
    # # plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    #
    plt.show()
    masks_comb_uint8 = masks_comb.astype(np.uint8)
    seg_filename = os.path.join(mask_out_dir, f"frame_{count:05d}_seg.png")
    imageio.imwrite(seg_filename, masks_comb_uint8)


