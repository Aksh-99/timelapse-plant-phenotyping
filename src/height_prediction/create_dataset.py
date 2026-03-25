import os
import csv
import shutil
import requests
from datetime import date, timedelta
from collections import Counter

# ─── PATHS ─────────────────────────────────────────────
LABEL_TRAIN_DIR = "data/detection/labels/train"
OUTPUT_CSV = "data/predictions/height_dataset.csv"
FRONTEND_CSV = "frontend/public/height_dataset.csv"

# ─── WEATHER API ──────────────────────────────────────
def get_temperature_for_date(target_date):
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude=40.7&longitude=-74.0"
            f"&daily=temperature_2m_max"
            f"&start_date={target_date}&end_date={target_date}"
            "&temperature_unit=fahrenheit"
            "&timezone=auto"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["daily"]["temperature_2m_max"][0]
    except Exception as e:
        print(f"Weather fetch failed for {target_date}: {e}")
        return 40.0  # fallback temp in F

# ─── STAGE DETECTION FROM A DAY FOLDER ────────────────
def get_stage_from_labels(day_folder):
    if not os.path.exists(day_folder) or not os.path.isdir(day_folder):
        return 0

    label_files = sorted(
        f for f in os.listdir(day_folder)
        if f.endswith(".txt")
    )

    if not label_files:
        return 0

    classes = []

    for label_file in label_files:
        file_path = os.path.join(day_folder, label_file)

        try:
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    classes.append(cls)
        except Exception as e:
            print(f"Skipping bad label file {file_path}: {e}")

    if not classes:
        return 0

    most_common = Counter(classes).most_common(1)[0][0]
    return most_common

# ─── COPY CSV TO FRONTEND ─────────────────────────────
def copy_csv_to_public():
    os.makedirs(os.path.dirname(FRONTEND_CSV), exist_ok=True)
    shutil.copy(OUTPUT_CSV, FRONTEND_CSV)
    print(f"Copied CSV to frontend: {FRONTEND_CSV}")

# ─── MAIN DATASET CREATION ────────────────────────────
def create_dataset():
    if not os.path.exists(LABEL_TRAIN_DIR):
        raise FileNotFoundError(f"Labels folder not found: {LABEL_TRAIN_DIR}")

    day_folders = sorted(
        f for f in os.listdir(LABEL_TRAIN_DIR)
        if os.path.isdir(os.path.join(LABEL_TRAIN_DIR, f))
    )

    if not day_folders:
        raise ValueError(f"No day folders found in: {LABEL_TRAIN_DIR}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    start_date = date(2026, 3, 12)

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "day",
            "folder_name",
            "date",
            "temperature",
            "stage_label",
            "height"
        ])

        for i, folder_name in enumerate(day_folders):
            day_num = i + 1
            current_date = start_date + timedelta(days=i)

            day_folder_path = os.path.join(LABEL_TRAIN_DIR, folder_name)
            stage_label = get_stage_from_labels(day_folder_path)
            temperature = get_temperature_for_date(current_date.isoformat())
            height = 0.0  # placeholder for now

            writer.writerow([
                day_num,
                folder_name,
                current_date.isoformat(),
                round(temperature, 2),
                stage_label,
                height
            ])

            print(
                f"Processed {folder_name}: "
                f"Day={day_num}, Stage={stage_label}, Temp={round(temperature, 2)}"
            )

    print(f"\nSaved CSV to: {OUTPUT_CSV}")
    copy_csv_to_public()

# ─── RUN ──────────────────────────────────────────────
if __name__ == "__main__":
    create_dataset()