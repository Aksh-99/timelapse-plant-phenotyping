import os
import pandas as pd

FRAMES_ROOT = "data/frames"
OUTPUT_CSV = "data/labels/dataset.csv"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_day_number(folder_name):
    # "day_01" -> 1
    return int(folder_name.replace("day_", ""))


def main():
    rows = []

    for day_folder in sorted(os.listdir(FRAMES_ROOT)):
        day_folder_path = os.path.join(FRAMES_ROOT, day_folder)

        if not os.path.isdir(day_folder_path):
            continue

        if not day_folder.startswith("day_"):
            continue

        day_number = get_day_number(day_folder)

        image_files = sorted(
            [
                f for f in os.listdir(day_folder_path)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ]
        )

        sampled_images = image_files[::10]

        for image_file in sampled_images:
            image_path = os.path.join(day_folder_path, image_file)

            rows.append({
                "image_path": image_path,
                "day": day_number,
                "height_cm": "",
                "stage": ""
            })

    df = pd.DataFrame(rows)
    os.makedirs("data/labels", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Created {OUTPUT_CSV} with {len(df)} rows.")


if __name__ == "__main__":
    main()