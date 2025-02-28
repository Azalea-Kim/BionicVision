# run this
# this needs new environment with tensorflow

import os
import cv2
import numpy as np
from timeit import default_timer as timer
from deepgaze.saliency_map import FasaSaliencyMapping


def process_images(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files from the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping invalid image: {image_file}")
            continue

        # Initialize saliency object for the image
        my_map = FasaSaliencyMapping(image.shape[0], image.shape[1])

        start = timer()
        saliency_map = my_map.returnMask(image, tot_bins=8, format='BGR2LAB')
        saliency_map = cv2.GaussianBlur(saliency_map, (3, 3), 1)  # Apply Gaussian blur to smooth it
        end = timer()
        print(f"--- {end - start} seconds for {image_file} ---")

        # Save the saliency map to the output folder
        output_path = os.path.join(output_folder, f"saliency_{image_file}")
        cv2.imwrite(output_path, saliency_map)
        print(f"Saved saliency map for {image_file} to {output_path}")


def main():
    input_folder = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\output_frames\\kitchen20fps"  # Specify the folder containing input images
    output_folder = "D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\saliency_output\\kitchen20fps"  # Specify the folder where saliency maps will be saved

    process_images(input_folder, output_folder)


if __name__ == "__main__":
    main()
