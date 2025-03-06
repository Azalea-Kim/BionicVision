import glob, os, cv2, shutil
import matplotlib.pyplot as plt
import numpy as np
import moviepy.editor as mp
from PIL import Image
local_path = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master"
depth_path = local_path+"\\depth_output_npy\\kitchen20fps_monodepth2_frames.mp4"
seg_path = local_path+"\\segmentation_output\\segmentation_video_kitchen_20fps.mp4"
sal_path = local_path+"\\saliency_output\\saliency_video_kitchen_20fps.mp4"
sal_path = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\saliency3\\saliency_DG3_4fixation.mp4"
comb_path = local_path+"\\combination_output\\"

video = "deepgaze3-4-75"
# threshold =

print(video)
seg_v = cv2.VideoCapture(seg_path)
sal_v = cv2.VideoCapture(sal_path)
dep_v = cv2.VideoCapture(depth_path)


def print_video_size(cap, video_name):
    if not cap.isOpened():
        print(f"{video_name} could not be opened.")
        return
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"{video_name} size: {int(width)}x{int(height)}")

# print_video_size(seg_v, "seg")
# print_video_size(sal_v, "sal")
# print_video_size(dep_v, "dep")


fps = 20
size = (1920, 1440)
# vid_pathOut = comb_path + "/%s.avi" % video
# out = cv2.VideoWriter(vid_pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
vid_pathOut = comb_path + "/%s.mp4" % video  # Change extension to .mp4

# Use 'mp4v' codec for MP4 output
out = cv2.VideoWriter(vid_pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)


success1 = 1
success2 = 1
success3 = 1

while success1 and success2 and success3:
    success1, seg = seg_v.read()
    success2, sal = sal_v.read()
    success3, dep = dep_v.read()

    if success1 == 0 or success2 == 0 or success3 == 0:
        break

    sal_fil = sal.copy()
    threshold = np.max(sal_fil) * .75  #.90
    # print(threshold)
    sal_fil[sal_fil <= threshold] = 0
    sal_fil[sal_fil > 0] = 255

    # normalize between [0,1]
    sal_norm = cv2.normalize(sal_fil, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    seg_norm = cv2.normalize(seg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dep_norm = cv2.normalize(dep, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    seg_norm = cv2.resize(seg_norm, (sal_norm.shape[1], sal_norm.shape[0]))
    dep_norm = cv2.resize(dep_norm, (sal_norm.shape[1], sal_norm.shape[0]))
    seg_sal = np.max((sal_norm[:, :, 0], seg_norm[:, :, 0]), axis=0)
    dep_seg_sal = dep_norm[:, :, 0].copy()
    dep_seg_sal[seg_sal == 0] = 0

    # result = dep_seg_sal.copy() * 255
    # result = seg_sal.copy()*255
    result = sal_norm.copy()*255
    out.write(np.uint8(result))


    # result3 = cv2.merge([result, result, result])
    # out.write(np.uint8(result3))

out.release()
cv2.destroyAllWindows()


