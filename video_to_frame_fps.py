import cv2
import os

def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count >= 400 and frame_count <= 1000:
            saved_frame = frame_count - 400
            cv2.imwrite(os.path.join(output_folder, f'{saved_frame:05d}_img.jpg'), frame)
        frame_count += 1
    cap.release()

    print(f"Successfully saved frames to {output_folder}。")

video_path = ''
output_folder = ''
video_to_frames(video_path, output_folder)