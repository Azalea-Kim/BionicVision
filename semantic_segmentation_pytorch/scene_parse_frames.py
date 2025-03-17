"""
Author: Yanxiu Jin
Date: 2025-03-17
Description: Description: Object segmentation script for MIT Scene parse implemented from baseline source code
"""
# System libs
# categories:
# https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit?pli=1&gid=0#gid=0
import os, csv, torch, scipy.io, torchvision.transforms, glob, cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from skimage import morphology

# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

all_frames = glob.glob("D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\output_frames\\table_try\\*.jpg")
for count in np.arange(1, len(all_frames)):  # each frame
    # f_name = "\\frame%d.jpg" % count
    f_name = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\output_frames\\kitchen_try\\frame_%03d.jpg" % count

    colors = scipy.io.loadmat('data/color150.mat')['colors']
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]


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


    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
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

    visualize_result(img_original, pred)

    # # Top classes in answer
    # predicted_classes = np.bincount(pred.flatten()).argsort()[::-1]
    # for c in predicted_classes[:15]:
    #     class_id = c + 1  # Ensure it matches 1-indexed labels
    #     class_name = names.get(class_id, "Unknown Class")
    #     print(f"Class {class_id}: {class_name}")
    #     if class_id in [1, 15]:  # wall door
    #         visualize_result(img_original, pred, c)

    '''
      1 wall
      4 floor
      8 bed
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


    classes = [0, 14]  # wall, door [1-1, 15-1] index
    pred_clean = pred.copy()
    # print(np.unique(pred_clean))

    pred_clean[~np.isin(pred_clean, classes)] = -1
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
    # print(image_edge.shape)
    # plt.imshow(image_edge, cmap='gray')
    # plt.title('4-')
    # plt.show()

    minLineLength = 30
    maxLineGap = 1
    lines = cv2.HoughLinesP(image_edge, 1, np.pi / 180, 15, minLineLength=minLineLength, maxLineGap=maxLineGap)
    edges = np.zeros(pred_clean3.shape) # initialize a blank image
    try: #modified here
        height, width = edges.shape
        border_threshold = 10
        min_length = 5  # define noise

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
