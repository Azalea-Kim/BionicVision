import cv2
import os

def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        print(frame_count)
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(output_folder, f'{frame_count:05d}.jpg'), frame)
        frame_count += 1
    cap.release()

    print(f"Successfully saved frames to {output_folder}。")

video_path = r'./data/kitchen.mp4'
output_folder = 'output_frames'
video_to_frames(video_path, output_folder)