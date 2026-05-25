"""
Author: Yanxiu Jin
Date: 2025-03-17
Description: Takes a video and output frames based on passed fps value
"""

def video_to_frames(video_path, output_folder, target_fps=20):
    video_capture = cv2.VideoCapture(video_path)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = video_capture.get(cv2.CAP_PROP_FPS)

    frame_interval = int(fps // target_fps)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    while True:
        ret, frame = video_capture.read()

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count + 1:03d}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    video_capture.release()
    print(f"Successfully saved {frame_count // frame_interval} frames to {output_folder}。")


video_path = 'D:\\2021-han-scene-simplification-master\\2021-han-scene-simplification-master\\ego4Ddata\\v2\\full_scale\\kitchen.mp4'  # 替换为视频文件路径
output_folder = 'output_frames\\kitchen20fps'
video_to_frames(video_path, output_folder, target_fps=20)
