import os
import cv2

input_folder = "data/raw_videos"
output_root = "data/frames"

os.makedirs(output_root, exist_ok=True)

video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]

print("Found videos:", video_files)

for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)

    video_name = os.path.splitext(video_file)[0]
    output_folder = os.path.join(output_root, video_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing: {video_path}")

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Could not open {video_path}")
        continue

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Saved {saved_count} frames for {video_file}")

cv2.destroyAllWindows()
print("Finished extracting frames from all videos.")