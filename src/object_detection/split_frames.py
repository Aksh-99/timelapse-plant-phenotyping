import os
import shutil
import random

FRAMES_ROOT = "data/frames"
TRAIN_ROOT = "data/detection/images/train"
VAL_ROOT = "data/detection/images/val"


def get_image_files(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def split_day_folder(day_folder):
    day_name = os.path.basename(day_folder)

    train_day = os.path.join(TRAIN_ROOT, day_name)
    val_day = os.path.join(VAL_ROOT, day_name)

    os.makedirs(train_day, exist_ok=True)
    os.makedirs(val_day, exist_ok=True)

    images = get_image_files(day_folder)

    if len(images) == 0:
        print(f"Skipping {day_name} (no images)")
        return

    random.shuffle(images)

    half = len(images) // 2

    train_images = images[:half]
    val_images = images[half:]

    for img in train_images:
        src = os.path.join(day_folder, img)
        dst = os.path.join(train_day, img)
        shutil.copy(src, dst)

    for img in val_images:
        src = os.path.join(day_folder, img)
        dst = os.path.join(val_day, img)
        shutil.copy(src, dst)

    print(f"{day_name}: {len(train_images)} train, {len(val_images)} val")


def main():

    if not os.path.exists(FRAMES_ROOT):
        print("Frames folder not found")
        return

    day_folders = [
        os.path.join(FRAMES_ROOT, d)
        for d in os.listdir(FRAMES_ROOT)
        if d.startswith("day_")
    ]

    day_folders.sort()

    for day in day_folders:
        split_day_folder(day)

    print("\nFinished splitting frames for detection.")


if __name__ == "__main__":
    main()