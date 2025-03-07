import os
import cv2
import imageio
import numpy as np


def create_video_from_frames(input_folder, output_video_path, frame_rate=20):
    frames = []  # Store saliency maps of each frame

    # List all files in the folder and sort them if necessary
    file_names = sorted(os.listdir(input_folder))  # Assumes the filenames are in a numerical order (e.g., frame1.jpg, frame2.jpg)

    for fname in file_names:
        # Construct full file path
        frame_path = os.path.join(input_folder, fname)

        # Check if the file is a valid image file (you can add more formats if needed)
        if frame_path.endswith('.jpg') or frame_path.endswith('.png') or frame_path.endswith('.jpeg'):
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale for saliency maps
            if frame is not None:
                frames.append(frame)
                print(f"Processed frame {fname}")
            else:
                print(f"Failed to load {frame_path}")

    # Create a writer object with imageio
    with imageio.get_writer(output_video_path, fps=frame_rate) as writer:
        # Write each frame to the video
        for frame in frames:
            writer.append_data(frame)  # Append the saliency map as a frame
        print(f"Video saved to {output_video_path}")


input_folder = 'C:/Users/ronki/OneDrive/Documents/GitHub/BionicVision/data/saliency_output'
output_video_path = 'C:/Users/ronki/OneDrive/Documents/GitHub/BionicVision/data/saliency_output/kitchen.mp4'

create_video_from_frames(input_folder, output_video_path)
