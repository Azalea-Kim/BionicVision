# categories:
# https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit?pli=1&gid=0#gid=0
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

# Get absolute path to the project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "semantic_segmentation_pytorch"))


# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

# For detectron2
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import json,random

sys.path.append(os.path.join(project_root, "detectron2-main"))

# Import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



# Load detectron2 model
cfg = get_cfg()
# Load model config file
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Set confidence threshold for predictions
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Load pre-trained model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Check COCO dataset metadata
metadata = MetadataCatalog.get("coco_2017_train")  # Use a predefined dataset from Detectron2
print(metadata)



# Load segmentation model
local_dir = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master"
colors = scipy.io.loadmat(local_dir+'\\semantic_segmentation_pytorch\\data\\color150.mat')['colors']
names = {}
with open(local_dir+'\\semantic_segmentation_pytorch\\data\\object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

# Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights=local_dir + '\\semantic_segmentation_pytorch\\ckpt\\ade20k-resnet50dilated-ppm_deepsup\\encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights=local_dir + '\\semantic_segmentation_pytorch\\ckpt\\ade20k-resnet50dilated-ppm_deepsup\\decoder_epoch_20.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()

# Load and normalize one image as a singleton tensor batch
pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])

def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index + 1]}:')

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)
    Image.fromarray(im_vis).show()  # Changed

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


def adaptive_circle_mask(
    mask_image,
    hand_x,
    hand_y,
    r_min,
    r_max) :

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

    # 5. 线性反比计算 ratio
    ratio = 1 - d / d_max  # 距离越大 ratio 越小
    ratio = max(0, min(1, ratio))  # 限制在 [0,1] 范围

    # 6. 根据 ratio 计算最终半径
    radius = int(r_min + (r_max - r_min) * ratio)
    #
    # # 7. 生成圆形掩膜
    # circle_mask = np.zeros((h, w), dtype=np.uint8)
    # cv2.circle(circle_mask, (int(hand_x), int(hand_y)), radius, 255, -1)

    return radius

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


def plot_adaptive_circles(mask_image,
    coords_list,
    r_min,
    r_max, isLinear):
    """
    在 mask_image 上，针对 coords_list 中的每个 (hand_x, hand_y)，
    计算自适应圆半径，并用 Matplotlib 画不填充的圆。
    """
    # 1. 建立图像显示
    fig, ax = plt.subplots(figsize=(8, 6))
    background = np.zeros_like(mask_image)
    ax.imshow(background, cmap='gray')
    # # 如果是灰度图，用 cmap='gray'
    # if len(mask_image.shape) == 2:
    #     ax.imshow(mask_image, cmap='gray')
    # else:
    #     # 如果是 BGR 彩色图，需要转为 RGB
    #     img_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    #     ax.imshow(img_rgb)



    # 2. 对 coords_list 里的每个坐标画一个圆
    for (hand_x, hand_y) in coords_list:
        # 计算自适应半径
        if isLinear:
            radius = adaptive_circle_mask(mask_image, hand_x, hand_y, r_min, r_max)
            ax.set_title("Adaptive Circles for Different Hand Positions (Linear), [100,400]")
            ax.axis("off")
        else:
            radius = exponential_circle_mask(mask_image,hand_x, hand_y, r_min, r_max, 2.0, True)
            ax.set_title("Adaptive Circles for Different Hand Positions (Exponential), [50,400], alpha=2.0, squared ratio")
            ax.axis("off")

        # 画一个不填充的圆 (fill=False)
        circle = Circle(
            (hand_x, hand_y),  # 圆心
            radius,
            fill=False,
            color='red',
            linewidth=2
        )
        ax.plot(hand_x, hand_y, marker='o', color='blue', markersize=5)
        ax.add_patch(circle)

    plt.show()

def generate_grid_coords(mask_image, nx=6, ny=5, margin=50):
    h, w = mask_image.shape[:2]

    # 如果只有1个点，间隔就设为0，避免除以 (nx-1)=0 的错误
    x_space = (w - 2 * margin) / (nx - 1) if nx > 1 else 0
    y_space = (h - 2 * margin) / (ny - 1) if ny > 1 else 0

    coords = []
    for j in range(ny):
        for i in range(nx):
            x = margin + i * x_space
            y = margin + j * y_space
            coords.append((x, y))
    return coords

def intersection_percentage(object_mask: np.ndarray, circle_mask: np.ndarray) -> float:
    """
    Computes the percentage of the object_mask (0/255) that intersects with the circle_mask (0/255).
    """
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



### Start from scene segmentation
# output_frames_dir = local_dir+"\\output_frames\\kitchen20fps"
# output_frames_dir = local_dir+"\\output_frames\\kitchen20_detect_try"
output_frames_dir = local_dir+"\\output_frames\\kitchen20_scene_try"
# output_frames_dir = local_dir+"\\output_frames\\kitchen20_personhistory_try"
all_frames = glob.glob(output_frames_dir+"\\*.jpg")


# Deal with cases with error in hand detection
person_history = deque([False, False, False, False, False], maxlen=5)
most_recent_circle_mask = None

# Get bright if no hand is detected for long
no_hand_frames = 0
no_hand_threshold = 30


W = 10  # store the most recent W frames' edge detection result
w_count = 0
edge_rep = np.zeros((1440, 1920, W))



for count in np.arange(1, len(all_frames)+1 ):  # each frame !!modified +1
    # f_name = "\\frame%d.jpg" % count
    f_name = output_frames_dir+"\\frame_%03d.jpg" % count
    # f_name = "input.jpg"
    print("processing scene segmentation...")
    # pil_image = Image.open('ADE_val_00001519.jpg').convert('RGB')
    pil_image = Image.open(f_name).convert('RGB')

    img_original = np.array(pil_image)

    height, width = img_original.shape[:2]

    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]

    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()



    # !!!一会放出来
    visualize_result(img_original, pred)

    # # Top classes in answer
    # predicted_classes = np.bincount(pred.flatten()).argsort()[::-1]
    # for c in predicted_classes[:15]:
    #     class_id = c + 1  # Ensure it matches 1-indexed labels
    #     class_name = names.get(class_id, "Unknown Class")
    #     print(f"Class {class_id}: {class_name}")
    #     if class_id in [1, 15]:  # wall door
    #         visualize_result(img_original, pred, c)

    # scene 和 object不同亮度
    '''
      1 wall
      4 floor
      8 bed
      9 window
      11 cabinent
      15 door
      16 table
      20 chair
      24 sofa
      25 shelf
      34 desk
      38 bathtub
      42 box
      48 sink
      52 refreigerator
      54 stairs
      72 stove
      125 microwave

    '''
    # classes = [0, 14]  # wall, door [1-1, 15-1] index
    classes = [8, 14]  # window, door [1-1, 15-1] index
    pred_clean = pred.copy()
    # print(np.unique(pred_clean))

    pred_clean[~np.isin(pred_clean, classes)] = -1 # modified!!! -1
    # print(np.unique(pred_clean))
    # plt.imshow(pred_clean, cmap='gray')
    # plt.title('1')
    # plt.show()

    # filter out small islands
    pred_clean2 = morphology.remove_small_objects(pred_clean.astype(bool), min_size=16000).astype(int) * 255

    # print(np.unique(pred_clean2))
    # plt.imshow(pred_clean2, cmap='gray')
    # plt.title('2')
    # plt.show()

    # combine mask with correct class labels
    pred_clean3 = np.minimum(pred_clean, pred_clean2)
    # plt.imshow(pred_clean3, cmap='gray')
    # plt.title('3')
    # plt.show()

    # np.savetxt("D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\semantic-segmentation-pytorch\\pred_cleans\\pred_clean.txt", pred_clean, fmt="%d")
    # np.savetxt("D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\semantic-segmentation-pytorch\\pred_cleans\\pred_clean2.txt", pred_clean2, fmt="%d")
    # np.savetxt("D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\semantic-segmentation-pytorch\\pred_cleans\\pred_clean3.txt", pred_clean3, fmt="%d")


    # get structure edges and get only long ones
    image = Image.fromarray(np.uint8((pred_clean3+1) * 255), 'L') # turn to black and white
    image_edge = image.filter(ImageFilter.FIND_EDGES) # edge detection
    image_edge = np.array(image_edge)
    kernel = np.ones((10, 10), np.uint8) # unit8 [0,255]
    image_edge = cv2.dilate(image_edge, kernel, iterations=1) # make edges thicker and continuous
    print(image_edge.shape)
    # plt.imshow(image_edge, cmap='gray')
    # plt.title('4-')
    # plt.show()

    edges_uint8 = image_edge.astype(np.uint8)

    # Here edges are stil connected
    # change later
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_uint8, connectivity=8)
    # area_threshold = 550  # input
    area_threshold = 1500
    edges_filtered = np.zeros_like(edges_uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            edges_filtered[labels == label] = 255

    # plt.imshow(edges_filtered, cmap="gray")
    # plt.title("Filtered Edges")
    # plt.axis("off")
    # plt.show()

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
    # plt.imshow(edges, cmap='gray')
    # plt.title(str(count)+'5')
    # plt.show()

    kernel = np.ones((10, 10), np.uint8)
    # erode to reduce noise
    edges = cv2.erode(get_houghlines(edges), kernel)
    # plt.title('6')
    # plt.imshow(edges, cmap='gray')
    # plt.show()

    edges_uint8 = edges.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_uint8, connectivity=8)
    area_threshold = 1500
    edges_filtered = np.zeros_like(edges_uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            edges_filtered[labels == label] = 255

    plt.imshow(edges_filtered, cmap="gray")
    plt.title("Filtered Edges")
    plt.axis("off")
    plt.show()

    edges = edges_filtered

    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # # ???? erode or dilate won't create non-binary values because (255,255,255) was the only process before
    # # !!! modified So I don't think we need threshold here:
    #
    # # (thresh, binRed) = cv2.threshold(hist_curr, 0, 255, cv2.THRESH_BINARY)
    #
    # # erode is just shrinking white regions
    # # morphologyEx can reduce smaller noise, make edges smoother
    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel2, iterations=3)
    # plt.title('7')
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    # # hough and erode strenghthen edges structures and reduce noise
    # edges = cv2.erode(get_houghlines(edges), kernel)
    # plt.title('8')
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    #

    #
    #
    #
    # if count <= W: # count is current frame index
    #     edge_rep[:, :, count - 1] = edges
    # else:
    #     # no big differnece...., even add noise
    #     # if we already have at least 10 frames
    #     # update current edge
    #
    #     # turn current frame edges into (height, width, 1)
    #     # concatenate in time dimension to (height, width, W+1)
    #     hist_curr = np.concatenate([edge_rep, np.expand_dims(edges, 2)], axis=2)
    #     # store all existed edges together within W frames
    #     hist_curr = np.max(hist_curr, axis=2)   # !! Need to change to average. Lots of Artifacts.
    #     plt.imshow(hist_curr)
    #
    #     # erode to reduce noise
    #     hist_curr = cv2.erode(get_houghlines(hist_curr), np.ones((10, 10)))
    #     plt.imshow(hist_curr)
    #
    #
    #     kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    #     # ???? erode or dilate won't create non-binary values because (255,255,255) was the only process before
    #     # So I don't think we need threshold here:
    #
    #     # (thresh, binRed) = cv2.threshold(hist_curr, 0, 255, cv2.THRESH_BINARY)
    #
    #     # erode is just shrinking white regions
    #     # morphologyEx can reduce smaller noise, make edges smoother
    #     hist_curr = cv2.morphologyEx(hist_curr, cv2.MORPH_OPEN, kernel2, iterations=3)
    #
    #     # hough and erode strenghthen edges structures and reduce noise
    #     hist_curr = cv2.erode(get_houghlines(hist_curr), np.ones((10, 10)))
    #     edges = hist_curr
    #     # plt.imshow(edges, cmap='gray')
    #     # plt.title(str(count)+'9')
    #     # plt.show()

    ####### Start Object segmentation with detectron2

    print("processing object segmentation...")
    # Read image
    im = cv2.imread(f_name)

    # Run Detectron2 predictor
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    instances = outputs["instances"]

    # visualize input
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])  # get COCO data
    visualizer = Visualizer(im, metadata=metadata, scale=1.2)
    vis_output = visualizer.draw_instance_predictions(instances.to("cpu"))

    # Get class indices and class names
    class_indices = instances.pred_classes.tolist()  # List of class indices
    class_names = metadata.thing_classes  # List of class names

    # # Print class index and class name for each detected instance
    # for i, class_idx in enumerate(class_indices):
    #     print(f"Instance {i}: Class Index = {class_idx}, Class Name = {class_names[class_idx]}")

    # Define important object classes (COCO class IDs)
    important_classes = [39, 41, 42, 44, 45, 63, 43, 0, 69]  # need to -1 for index
        # bottle, cup, fork, spoon, bowl, laptop, knife, person, oven

    # 78 hair drier;
    min_area = 80000
    is_person = False


    # Check if instance has predicted classes
    if instances.has("pred_classes"):
        # classes = instances.pred_classes.cpu().numpy()
        classes = np.asarray(instances.pred_classes.cpu().numpy())
        # print(cla.shape)
        # print(cla) # [45 39  0 39 69 39 39 78 39 43 44 39 73 41 39 39]
        # print(cla[0])
    else:
        classes = None

    # Get instance masks if available
    # get segmentation mask (N, H, W)
    if instances.has("pred_masks"):
        masks = np.asarray(instances.pred_masks.cpu().numpy())

        # print(masks.shape) #(16, 1440, 1920)
    else:
        masks = None

    # Filter out not important classes
    classes_fil = []
    if classes is not None:
        for c in classes:
            if c in important_classes:
                classes_fil.append(1)
                if c == 0:
                    is_person = True
                    # [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
            else:
                classes_fil.append(0)

    # Determine final mask
    if np.sum(classes_fil) == 0:  # No important objects detected  ###! Maybe show less important objects...
        # Only show scene edges from previous segmentation
        masks_comb = edges  # `edges` needs to be defined elsewhere
    else:
        masks_idx = np.where(np.array(classes_fil) == 1)[0]
        # print(classes_fil) # [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
        # print(masks_idx) # [ 0  1  2  3  4  5  6  8  9 10 11 13 14 15]

        # Find hand region

        # // limitation is here it does not segment hand but person so if other person is also highlighted
        hand_mask = np.zeros_like(masks[0])
        # either let it brighten the whole object or just the intersecting part
        circle_mask = np.zeros_like(masks[0])

        if is_person:
            for i in masks_idx:
                if classes[i] == 0:
                    mask = masks[i]
                    mask_image1 = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_image = mask*255

                    # plt.figure(figsize=(10, 10))
                    # plt.imshow(mask_image1)
                    # plt.title("person_mask")
                    # plt.axis("off")
                    # plt.show()

                    # unique_values = np.unique(mask)
                    # print("Unique values in mask:", unique_values)

                    hand_mask = np.maximum(hand_mask, mask * 255)  # 单独分出来后面可以少加一次

                    ys, xs = np.where(mask_image == 255)
                    # Safety check in case the mask is empty
                    if len(xs) == 0 or len(ys) == 0:
                        raise ValueError("Mask appears empty or no 255 region found!")

                    y_min, y_max = np.min(ys), np.max(ys)
                    x_min, x_max = np.min(xs), np.max(xs)

                    width = x_max - x_min
                    height = y_max - y_min

                    radius = np.sqrt(width**2 + height**2) / 2.0

                    # print(radius) #220 #360
                    # centroid_x = np.mean(xs)
                    # centroid_y = np.mean(ys)

                    angle_degrees = PCA_get_angle(xs,ys,False)

                    # 左上 -156 负
                    # 右上 111 正
                    if angle_degrees < 0:
                        centroid_x, centroid_y = x_min, y_min
                    elif angle_degrees > 0:
                        centroid_x, centroid_y = x_max, y_min
                    else:
                        centroid_x, centroid_y = (x_min+x_max)/2, y_min

                    # #radius experiments
                    # # 随机设置一些 (hand_x, hand_y) 坐标
                    # coords = generate_grid_coords(mask_image, nx=5, ny=5, margin=50)
                    # # 调用绘制函数
                    # plot_adaptive_circles(mask_image, coords, r_min=50, r_max=400, isLinear = False)


                    # get radius
                    # 直觉：越接近中心，表示用户更可能在专注于这部分区域，给一个更大的“可操作”半径。
                    # 越靠边缘，可能是手刚伸进画面，或者还未准备操作；可以减小半径避免“过度亮度”或“干扰”
                    # radius = adaptive_circle_mask(mask_image, centroid_x, centroid_y, 100,400)
                    radius = exponential_circle_mask(mask_image, centroid_x, centroid_y, 100 ,400, 2.0, True)

                    c_mask = np.zeros_like(circle_mask).astype(np.uint8)
                    # Draw an outline circle (not filled) with white=255
                    cv2.circle(c_mask, (int(centroid_x), int(centroid_y)), int(radius), 255, thickness=-1)
                    circle_mask = np.maximum(circle_mask, c_mask)
                    most_recent_circle_mask = circle_mask

                    # 在手上plot红色圆圈
                    # plt.figure(figsize=(8, 6))
                    # plt.imshow(mask_image,cmap="gray")
                    # plt.title("Circle on Hand/Arm Mask (Exponential Squared 2.0) [100,400]")
                    # plt.axis("off")
                    # circle = Circle(
                    #     (centroid_x, centroid_y),  # (x, y) in Matplotlib coordinates
                    #     radius,
                    #     color="red",
                    #     fill=False,  # Circle outline only
                    #     linewidth=2
                    # )
                    #
                    # # 8. Add the circle to the current axes
                    # ax = plt.gca()
                    # ax.add_patch(circle)
                    #
                    # # 9. Show the final result
                    # plt.show()


        # plt.figure(figsize=(10, 10))
        # plt.imshow(hand_mask, cmap="gray")
        # plt.title("Final hand masks")
        # plt.axis("off")
        # plt.show()
        # unique_values = np.unique(hand_mask)
        # print("Unique values in mask_edge:", unique_values)
        #
        # plt.figure(figsize=(10, 10))
        # plt.imshow(circle_mask, cmap="gray")
        # plt.title("Final circle masks")
        # plt.axis("off")
        # plt.show()
        # unique_values = np.unique(circle_mask)
        # print("Unique values in mask_edge:", unique_values)

        if len(masks_idx) > 0:
            # masks_fil = masks[masks_idx, :, :]
            # masks_comb = np.max(masks_fil, axis=0)
            masks_comb = np.zeros_like(masks[0])

            # print(masks_fil.shape) #(14, 1440, 1920)

            # class_idx = classes[masks_idx[i]] # new
            # print(class_idx)
            # !!! modified here if the detected important object is too big, set it to edges
            # for i, mask in enumerate(masks_fil):

            # print(person_history)
            for i, mask in enumerate(masks):
                if i in masks_idx and classes[i] != 0: # Important objects and no person
                    area = np.sum(mask)
                    ip=0.0
                    b = 160  # Base:b2
                    if is_person:
                        # print("有手")
                        ip = intersection_percentage(mask*255, circle_mask)
                    elif any(person_history):
                        # print("前两frame有手")
                        ip = intersection_percentage(mask * 255, most_recent_circle_mask)
                    # No hand detected for long time
                    elif no_hand_frames >= no_hand_threshold:
                        b = 220
                    # print (ip, classes[i])


                    if ip > 0.50:
                        b = 220 # b12 only hand circle intersect
                    if area > min_area:
                        # edge detection
                        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_edge = mask_image.filter(ImageFilter.FIND_EDGES)
                        mask_edge = np.array(mask_edge)

                        kernel = np.ones((20, 20), np.uint8)  # unit8 [0,255]
                        mask_edge = cv2.dilate(mask_edge, kernel, iterations=1)  # make edges thicker and continuous
                        # add edges of big objects


                        mask_edge = mask_edge.astype(np.float32)  # convert to float for scaling
                        mask_edge *= float(b) / 255.0  # scale to [0..200]
                        mask_edge = mask_edge.astype(np.uint8)  # convert back to uint8

                        masks_comb = np.maximum(masks_comb, mask_edge)
                        # print(b)
                    else:
                        # add normal mask
                        # masks_comb = np.maximum(masks_comb, mask * 255)
                        masks_comb = np.maximum(masks_comb, mask * b)
                        # print(b)
                    if is_person:
                        masks_comb = np.maximum(masks_comb, hand_mask)
        else:  # just in case masks_idx is empty
            masks_comb = np.zeros(im.shape[:2])  # Default to an empty mask
    # print(masks_comb.shape)
    # unique_values = np.unique(masks_comb)
    # print("Unique values in masks_comb:", list(unique_values))

    b = 160 # b2
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
    mask_out_dir = "segmentation_output/kitchen_20fps_prioritize_noscene_final"
    if not os.path.exists(mask_out_dir):
        os.mkdir(mask_out_dir)

    # Display and save segmented image
    print("Processing frame %d" % count)
    plt.imshow(masks_comb, cmap="gray",vmin=0, vmax=255)
    plt.axis("off")
    plt.title('Object segmentation (b2=160, b12=220, ip>50)')


    filename = "frame_%d_seg.jpg" % count
    filepath = os.path.join(mask_out_dir, filename)
    # plt.savefig(filepath, bbox_inches='tight', pad_inches=0)

    plt.show()
    # masks_comb_uint8 = masks_comb.astype(np.uint8)
    # seg_filename = os.path.join(mask_out_dir, f"frame_{count:03d}_seg.jpeg")
    # imageio.imwrite(seg_filename, masks_comb_uint8)


# 有个问题是，没有手的话就全部都是暗的 比如说31 32 （但是我觉得是model的问题，目前sota应该不会检测不到胳膊的存在）
# 用dequeue检测前5个frame如果有手，就用最近的那个circlemask


# 还有个问题，scene segmentation 比如windows 还有很多artifacts 也觉得是model的问题
