"""
Author: Yanxiu Jin
Date: 2025-03-17
Description: Object segmentation script for detectron2 implemented from baseline source code,
only retrieve important object classes
"""

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import common libraries
import numpy as np
import os
import json
import cv2
import random
import matplotlib.pyplot as plt  # Use Matplotlib to display images

# Import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
# Load model config file
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Set confidence threshold for predictions
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Load pre-trained model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Check COCO dataset metadata
metadata = MetadataCatalog.get("coco_2017_train")  # Use a predefined dataset from Detectron2
# print(metadata)

# Read image
im = cv2.imread("D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\detectron2-main\\input.jpg")

# Run Detectron2 predictor
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
instances = outputs["instances"]

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
visualizer = Visualizer(im, metadata=metadata, scale=1.2)
vis_output = visualizer.draw_instance_predictions(instances.to("cpu"))

plt.figure(figsize=(10, 10))
plt.imshow(vis_output.get_image())
plt.axis("off")
plt.show()



# Define important object classes (COCO class IDs)
important_classes = [0, 1, 2, 5, 7]  # person, bicycle, car, bus, train

# Check if instance has predicted classes
if instances.has("pred_classes"):
    classes = instances.pred_classes.cpu().numpy()
else:
    classes = None

# Get instance masks if available
if instances.has("pred_masks"):
    masks = np.asarray(instances.pred_masks.cpu().numpy())
else:
    masks = None

# Filter classes
classes_fil = []
if classes is not None:
    for c in classes:
        if c in important_classes:
            classes_fil.append(1)
        else:
            classes_fil.append(0)

# Determine final mask
if np.sum(classes_fil) == 0:  # No important objects
    masks_comb = edges  # `edges` needs to be defined elsewhere
else:
    masks_idx = np.where(np.array(classes_fil) == 1)[0]
    if len(masks_idx) > 0:
        masks_fil = masks[masks_idx, :, :]
        masks_comb = np.max(masks_fil, axis=0)
    else:
        masks_comb = np.zeros(im.shape[:2])  # Default to an empty mask

# Create output folder if it doesn't exist
if not os.path.exists("detectron_mask"):
    os.mkdir("detectron_mask")

# Display and save segmented image
print("Processing frame %d" % count)
plt.imshow(masks_comb, cmap="gray")
plt.axis("off")

filename = "frame_%d_seg.jpg" % count
filepath = os.path.join("detectron_mask", filename)
plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
plt.close()  # Close figure to prevent overlapping
