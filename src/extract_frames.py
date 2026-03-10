import os
import cv2


video_path = "data/raw_videos/day_01.mp4"
output_folder = "data/frames/day_01"
os.makedirs(output_folder, exist_ok=True)
video = cv2.VideoCapture(video_path)
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    #print("Reading frame...", frame)
    filename = f"{output_folder}/frame_{frame_count}.jpg"
    cv2.imwrite(filename, frame)
    frame_count += 1
    cv2.imshow('video',frame)
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()
