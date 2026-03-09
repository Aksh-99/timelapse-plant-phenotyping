import os
import cv2

os.makedirs("../data/frames", exist_ok=True)
video = cv2.VideoCapture('video_sample.mp4')
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    #print("Reading frame...", frame)
    filename = f"../data/frames/frame_{frame_count}.jpg"
    cv2.imwrite(filename, frame)
    frame_count += 1
    cv2.imshow('video',frame)
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()
