# categories:
# https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit?pli=1&gid=0#gid=0
import os, csv, torch, scipy.io, torchvision.transforms, glob, cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from skimage import morphology

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

### Start from scene segmentation


output_frames_dir = local_dir+"\\output_frames\\kitchen_try"
all_frames = glob.glob(output_frames_dir+"\\*.jpg")

W = 10 # store the most recent W frames' edge detection result
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
    print(np.unique(pred_clean))
    plt.imshow(pred_clean, cmap='gray')
    plt.title('1')
    plt.show()

    # filter out small islands
    pred_clean2 = morphology.remove_small_objects(pred_clean.astype(bool), min_size=16000).astype(int) * 255

    print(np.unique(pred_clean2))
    plt.imshow(pred_clean2, cmap='gray')
    plt.title('2')
    plt.show()

    # combine mask with correct class labels
    pred_clean3 = np.minimum(pred_clean, pred_clean2)
    plt.imshow(pred_clean3, cmap='gray')
    plt.title('3')
    plt.show()

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
    plt.imshow(image_edge, cmap='gray')
    plt.title('4-')
    plt.show()

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

    plt.imshow(edges_filtered, cmap="gray")
    plt.title("Filtered Edges")
    plt.axis("off")
    plt.show()

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
    plt.imshow(edges, cmap='gray')
    plt.title(str(count)+'5')
    plt.show()

    kernel = np.ones((10, 10), np.uint8)
    # erode to reduce noise
    edges = cv2.erode(get_houghlines(edges), kernel)
    plt.title('6')
    plt.imshow(edges, cmap='gray')
    plt.show()

    edges_uint8 = edges.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_uint8, connectivity=8)
    area_threshold = 1500
    edges_filtered = np.zeros_like(edges_uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            edges_filtered[labels == label] = 255

    plt.imshow(edges_filtered, cmap="gray")
    plt.title("Filtered Edges2")
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



    # Need to change this part to averaging and thresholding

    if count <= W: # count is current frame index
        edge_rep[:, :, count - 1] = edges
    else:
        # no big differnece...., even add noise
        # if we already have at least 10 frames
        # update current edge

        # turn current frame edges into (height, width, 1)
        # concatenate in time dimension to (height, width, W+1)
        hist_curr = np.concatenate([edge_rep, np.expand_dims(edges, 2)], axis=2)
        # store all existed edges together within W frames
        hist_curr = np.max(hist_curr, axis=2)
        plt.imshow(hist_curr)

        # erode to reduce noise
        hist_curr = cv2.erode(get_houghlines(hist_curr), np.ones((10, 10)))
        plt.imshow(hist_curr)


        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # ???? erode or dilate won't create non-binary values because (255,255,255) was the only process before
        # So I don't think we need threshold here:

        # (thresh, binRed) = cv2.threshold(hist_curr, 0, 255, cv2.THRESH_BINARY)

        # erode is just shrinking white regions
        # morphologyEx can reduce smaller noise, make edges smoother
        hist_curr = cv2.morphologyEx(hist_curr, cv2.MORPH_OPEN, kernel2, iterations=3)

        # hough and erode strenghthen edges structures and reduce noise
        hist_curr = cv2.erode(get_houghlines(hist_curr), np.ones((10, 10)))
        edges = hist_curr
        # plt.imshow(edges, cmap='gray')
        # plt.title(str(count)+'9')
        # plt.show()

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

    # Print class index and class name for each detected instance
    for i, class_idx in enumerate(class_indices):
        print(f"Instance {i}: Class Index = {class_idx}, Class Name = {class_names[class_idx]}")


    # visualize results
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_output.get_image())
    plt.axis("off")
    plt.show()


    # Define important object classes (COCO class IDs)
    important_classes = [39, 41, 42, 44, 45, 63, 43, 0, 69] # need to -1 for index
    # bottle, cup, fork, spoon, bowl, laptop, knife, person, oven
    min_area = 80000

    # Check if instance has predicted classes
    if instances.has("pred_classes"):
        classes = instances.pred_classes.cpu().numpy()
    else:
        classes = None

    # Get instance masks if available
    # get segmentation mask (N, H, W)
    if instances.has("pred_masks"):
        masks = np.asarray(instances.pred_masks.cpu().numpy())
    else:
        masks = None

    # Filter out not important classes
    classes_fil = []
    if classes is not None:
        for c in classes:
            if c in important_classes:
                classes_fil.append(1)
            else:
                classes_fil.append(0)

    # Determine final mask
    if np.sum(classes_fil) == 0:  # No important objects detected
        # Only show scene edges from previous segmentation
        masks_comb = edges  # `edges` needs to be defined elsewhere
    else:
        masks_idx = np.where(np.array(classes_fil) == 1)[0]
        if len(masks_idx) > 0:
            masks_fil = masks[masks_idx, :, :]
            # masks_comb = np.max(masks_fil, axis=0)
            masks_comb = np.zeros_like(masks[0])

            # !!! modified here if the detected important object is too big, set it to edges
            for i, mask in enumerate(masks_fil):
                area = np.sum(mask)

                if area > min_area:
                    # edge detection
                    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_edge = mask_image.filter(ImageFilter.FIND_EDGES)
                    mask_edge = np.array(mask_edge)

                    kernel = np.ones((20, 20), np.uint8)  # unit8 [0,255]
                    mask_edge = cv2.dilate(mask_edge, kernel, iterations=1)  # make edges thicker and continuous
                    # add edges of big objects
                    unique_values = np.unique(mask_edge)
                    print("Unique values in mask_edge:", unique_values)

                    masks_comb = np.maximum(masks_comb, mask_edge)
                else:
                    # add normal mask
                    masks_comb = np.maximum(masks_comb, mask*255)
        else: # just in case masks_idx is empty
            masks_comb = np.zeros(im.shape[:2])  # Default to an empty mask

    # Create output folder if it doesn't exist
    mask_out_dir = "detectron_mask_kitchen_try_2"
    if not os.path.exists(mask_out_dir):
        os.mkdir(mask_out_dir)

    # Display and save segmented image
    print("Processing frame %d" % count)
    plt.imshow(masks_comb, cmap="gray")
    plt.axis("off")
    plt.title('Object segmentation')


    filename = "frame_%d_seg.jpg" % count
    filepath = os.path.join(mask_out_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)

    plt.show()



