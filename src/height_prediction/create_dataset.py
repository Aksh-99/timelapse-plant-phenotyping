import os
import csv
import shutil
import requests
from datetime import date, timedelta
from collections import Counter
 
# ─── PATHS ───────────────────────────────────────────────────────────────────
LABEL_TRAIN_DIR = "data/detection/labels/train"
OUTPUT_CSV      = "data/predictions/height_dataset.csv"
FRONTEND_CSV    = "frontend/public/height_dataset.csv"
 
 
# ─── WEATHER API ─────────────────────────────────────────────────────────────
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
        print(f"  Weather fetch failed for {target_date}: {e}")
        return 40.0
 
 
# ─── STAGE DETECTION ─────────────────────────────────────────────────────────
def get_stage_from_labels(day_folder):
    if not os.path.exists(day_folder) or not os.path.isdir(day_folder):
        return 0
 
    label_files = sorted(
        f for f in os.listdir(day_folder) if f.endswith(".txt")
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
                    if parts:
                        classes.append(int(parts[0]))
        except Exception as e:
            print(f"  Skipping bad label file {file_path}: {e}")
 
    if not classes:
        return 0
    return Counter(classes).most_common(1)[0][0]
 
 
# ─── LOAD EXISTING CSV ───────────────────────────────────────────────────────
def load_existing_heights(csv_path):
    """
    Returns a dict of { folder_name: height } for all rows
    that already have a non-zero height recorded.
    """
    existing = {}
    if not os.path.exists(csv_path):
        return existing
 
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                h = float(row["height"])
                if h > 0.0:
                    existing[row["folder_name"]] = h
            except (KeyError, ValueError):
                pass
 
    return existing
 
 
# ─── INTERACTIVE HEIGHT INPUT ────────────────────────────────────────────────
def prompt_height(folder_name, day_num, existing_height=None):
    """
    Ask the user to enter a height for this day.
    - If a height already exists, show it and allow keeping it (press Enter).
    - Accepts a float or blank (to keep existing / default to 0.0).
    """
    if existing_height is not None:
        prompt = (
            f"  Day {day_num:>3} | {folder_name} | "
            f"existing height = {existing_height:.4f} in "
            f"(press Enter to keep, or type new value): "
        )
    else:
        prompt = (
            f"  Day {day_num:>3} | {folder_name} | "
            f"enter height in inches (or press Enter to skip as 0.0): "
        )
 
    while True:
        raw = input(prompt).strip()
 
        if raw == "":
            # Keep existing or default to 0.0
            return existing_height if existing_height is not None else 0.0
 
        try:
            value = float(raw)
            if value < 0:
                print("  Height cannot be negative. Try again.")
                continue
            return value
        except ValueError:
            print("  Invalid input. Enter a number or press Enter to skip.")
 
 
# ─── COPY TO FRONTEND ────────────────────────────────────────────────────────
def copy_csv_to_public():
    os.makedirs(os.path.dirname(FRONTEND_CSV), exist_ok=True)
    shutil.copy(OUTPUT_CSV, FRONTEND_CSV)
    print(f"\nCopied CSV to frontend: {FRONTEND_CSV}")
 
 
# ─── MAIN ────────────────────────────────────────────────────────────────────
def create_dataset():
    if not os.path.exists(LABEL_TRAIN_DIR):
        raise FileNotFoundError(f"Labels folder not found: {LABEL_TRAIN_DIR}")
 
    day_folders = sorted(
        f for f in os.listdir(LABEL_TRAIN_DIR)
        if os.path.isdir(os.path.join(LABEL_TRAIN_DIR, f))
    )
    if not day_folders:
        raise ValueError(f"No day folders found in: {LABEL_TRAIN_DIR}")
 
    # ── Load any heights already saved ──────────────────────────────────────
    existing_heights = load_existing_heights(OUTPUT_CSV)
    preserved = len([h for h in existing_heights.values() if h > 0])
    print(f"\nFound {len(day_folders)} day folder(s).")
    print(f"Existing heights preserved: {preserved}\n")
 
    # ── Identify days that are new or still have height = 0 ─────────────────
    new_days = [
        f for f in day_folders if existing_heights.get(f, 0.0) == 0.0
    ]
 
    if not new_days:
        print("All days already have heights recorded.")
        print("To update a specific day, press Enter to keep or type a new value.\n")
 
    # ── Collect heights interactively ────────────────────────────────────────
    print("─" * 60)
    print("HEIGHT INPUT")
    print("─" * 60)
    print("Enter the measured height in inches for each day.")
    print("Press Enter to keep an existing value or skip a new one (saves as 0.0).\n")
 
    heights = {}
    start_date = date(2026, 3, 12)
 
    for i, folder_name in enumerate(day_folders):
        day_num = i + 1
        existing = existing_heights.get(folder_name)
 
        # Only prompt if new or has no height yet — skip already-measured days
        # unless the user runs with --all flag (see below)
        if existing is not None and existing > 0.0:
            heights[folder_name] = existing
            print(f"  Day {day_num:>3} | {folder_name} | kept: {existing:.4f} in")
        else:
            heights[folder_name] = prompt_height(folder_name, day_num, existing)
 
    # ── Write the full CSV ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("PROCESSING & SAVING")
    print("─" * 60)
 
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
 
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "day", "folder_name", "date",
            "temperature", "stage_label", "height"
        ])
 
        for i, folder_name in enumerate(day_folders):
            day_num     = i + 1
            current_date = start_date + timedelta(days=i)
            day_folder_path = os.path.join(LABEL_TRAIN_DIR, folder_name)
 
            stage_label = get_stage_from_labels(day_folder_path)
            temperature = get_temperature_for_date(current_date.isoformat())
            height      = heights.get(folder_name, 0.0)
 
            writer.writerow([
                day_num,
                folder_name,
                current_date.isoformat(),
                round(temperature, 2),
                stage_label,
                round(height, 4),
            ])
 
            print(
                f"  Day {day_num:>3} | {folder_name} | "
                f"Stage={stage_label} | Temp={round(temperature,2)} F | "
                f"Height={round(height,4)} in"
            )
 
    print(f"\nSaved CSV to: {OUTPUT_CSV}")
    copy_csv_to_public()
    print("\nDone.")
 
 
# ─── OPTIONAL: re-prompt all days including already-measured ones ─────────────
def update_all_heights():
    """
    Run this if you want to re-enter heights for every day,
    including ones already recorded. Existing values shown as defaults.
    """
    if not os.path.exists(LABEL_TRAIN_DIR):
        raise FileNotFoundError(f"Labels folder not found: {LABEL_TRAIN_DIR}")
 
    day_folders = sorted(
        f for f in os.listdir(LABEL_TRAIN_DIR)
        if os.path.isdir(os.path.join(LABEL_TRAIN_DIR, f))
    )
 
    existing_heights = load_existing_heights(OUTPUT_CSV)
 
    print("\n" + "─" * 60)
    print("UPDATE ALL HEIGHTS (press Enter to keep existing value)")
    print("─" * 60 + "\n")
 
    heights = {}
    for i, folder_name in enumerate(day_folders):
        day_num  = i + 1
        existing = existing_heights.get(folder_name)
        heights[folder_name] = prompt_height(folder_name, day_num, existing)
 
    # Re-use the same write logic
    start_date = date(2026, 3, 12)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
 
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "day", "folder_name", "date",
            "temperature", "stage_label", "height"
        ])
        for i, folder_name in enumerate(day_folders):
            day_num      = i + 1
            current_date = start_date + timedelta(days=i)
            day_folder_path = os.path.join(LABEL_TRAIN_DIR, folder_name)
            stage_label  = get_stage_from_labels(day_folder_path)
            temperature  = get_temperature_for_date(current_date.isoformat())
            height       = heights.get(folder_name, 0.0)
            writer.writerow([
                day_num, folder_name, current_date.isoformat(),
                round(temperature, 2), stage_label, round(height, 4),
            ])
 
    print(f"\nSaved to: {OUTPUT_CSV}")
    copy_csv_to_public()
    print("Done.")
 
 
# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
 
    if "--update-all" in sys.argv:
        update_all_heights()
    else:
        create_dataset()
 
