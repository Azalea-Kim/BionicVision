
import matplotlib.pyplot as plt

import cv2
import numpy as np

def create_circle_mask_from_sal(sal_fill):
    """
    sal_fill: a binary (0/255) mask with one or more white regions.

    Returns:
        circle_mask: a binary mask (same shape) with a filled circle (white=255)
                     that encloses the largest white region from sal_fill.
    """
    # 1. Ensure sal_fill is single-channel, uint8
    if len(sal_fill.shape) == 3 and sal_fill.shape[2] == 3:
        sal_fill = cv2.cvtColor(sal_fill, cv2.COLOR_BGR2GRAY)
    sal_fill = sal_fill.astype(np.uint8)

    # 2. Find contours
    contours, _ = cv2.findContours(sal_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No contours => return an empty mask
        return np.zeros_like(sal_fill, dtype=np.uint8)

    # 3. Pick the largest contour (by area)
    largest_contour = max(contours, key=cv2.contourArea)

    # If the contour is too small, just return empty
    if len(largest_contour) < 5:
        return np.zeros_like(sal_fill, dtype=np.uint8)

    # 4. Fit a minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

    # 5. Draw a filled circle on a blank mask
    circle_mask = np.zeros_like(sal_fill, dtype=np.uint8)
    cv2.circle(circle_mask, (int(cx), int(cy)), int(radius), 255, thickness=-1)

    return circle_mask

seg_img = cv2.imread("D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\segmentation_output\\kitchen_20fps_prioritize_noscene_final_png\\frame_163_seg.png",
                     cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
seg = np.uint8(seg_img)

sal_img = cv2.imread("D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\gaze_estimations\\kitchen_20fps\\gray\\gray_frame_163.jpg"
                     , cv2.IMREAD_GRAYSCALE)
sal = np.uint8(sal_img)


# Threshold the saliency map
sal_fil = sal.copy()
threshold = np.max(sal_fil) * .40  # .90
# threshold = np.percentile(sal, 90)
# print(threshold)
sal_fil[sal_fil <= threshold] = 0
sal_fil[sal_fil > 0] = 255


sal_fil1 = sal_fil
# Create the elliptical mask
sal_fil = create_circle_mask_from_sal(sal_fil)

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(sal_fil1, cmap="gray")
plt.title("Original Binary Mask (sal_fill)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sal_fil, cmap="gray")
plt.title("Fitted Ellipse Mask")
plt.axis("off")

plt.show()

seg_sal = np.maximum(seg,sal_fil)
seg_sal = np.where(seg == 0, seg, seg_sal)
#
# print("seg unique:", np.unique(seg))
# print("seg_sal unique:", np.unique(seg_sal))
#
#

intersection = np.where((seg > 0) & (sal_fil > 0), 1, 0).astype(np.uint8)
# If seg is already uint8 in [0..255], convert to 3-channel
seg_rgb = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)  # shape (H, W, 3)
# Mark intersection pixels in red (BGR: (0,0,255))


seg_rgb[intersection == 1] = (0, 0, 255)
plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(seg_rgb, cv2.COLOR_BGR2RGB))  # Convert BGR -> RGB for Matplotlib
plt.title("Intersection in Red")
plt.axis("off")
plt.show()


plt.imshow(seg_sal, cmap="gray", vmin=0, vmax=255)
plt.axis("off")
plt.title('seg_sal')
plt.show()