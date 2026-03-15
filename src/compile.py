import subprocess
import os
import sys


def run_step(step_name, command):
    print(f"\n=== {step_name} ===")
    print("Running:", " ".join(command))

    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"ERROR: {step_name} failed.")
        sys.exit(result.returncode)


def file_exists(path):
    return os.path.isfile(path)


def folder_has_files(folder):
    return os.path.isdir(folder) and any(
        os.path.isfile(os.path.join(folder, f)) for f in os.listdir(folder)
    )


def print_skipped(step_name, reason):
    print(f"\n=== {step_name} ===")
    print(f"Skipped: {reason}")


print("Starting full pipeline...")

# ==================================================
# STEP 1: EXTRACT FRAMES
# ==================================================
extract_frames_script = "src/data/extract_frames.py"

if file_exists(extract_frames_script):
    run_step("Step 1: Extracting frames", ["python3", extract_frames_script])
else:
    print_skipped("Step 1: Extracting frames", f"{extract_frames_script} not found")

# ==================================================
# STEP 2: CREATE DATASET CSV
# ==================================================
create_dataset_script = "src/height_prediction/create_dataset.py"

if file_exists(create_dataset_script):
    run_step("Step 2: Creating dataset CSV", ["python3", create_dataset_script])
else:
    print_skipped("Step 2: Creating dataset CSV", f"{create_dataset_script} not found")

# ==================================================
# STEP 3: TEST HEIGHT DATASET
# ==================================================
dataset_script = "src/height_prediction/dataset.py"

if file_exists(dataset_script):
    run_step("Step 3: Testing dataset", ["python3", dataset_script])
else:
    print_skipped("Step 3: Testing dataset", f"{dataset_script} not found")

# ==================================================
# STEP 4: PREPARE OBJECT DETECTION DATA
# ==================================================
train_detection_images = "data/detection/images/train"
val_detection_images = "data/detection/images/val"
train_detection_labels = "data/detection/labels/train"
val_detection_labels = "data/detection/labels/val"

split_frames_script = "src/object_detection/split_frames.py"

detection_data_ready = (
    folder_has_files(train_detection_images)
    and folder_has_files(val_detection_images)
    and folder_has_files(train_detection_labels)
    and folder_has_files(val_detection_labels)
)

if detection_data_ready:
    print_skipped(
        "Step 4: Splitting frames for detection",
        "detection image and label folders already contain files"
    )
else:
    if file_exists(split_frames_script):
        run_step(
            "Step 4: Splitting frames for detection",
            ["python3", split_frames_script]
        )
    else:
        print_skipped(
            "Step 4: Splitting frames for detection",
            f"{split_frames_script} not found"
        )

# ==================================================
# STEP 5: TRAIN SEED DETECTOR
# ==================================================
model_path = "models/seed_detector_fasterrcnn.pth"
train_detector_script = "src/object_detection/train_seed_detector.py"

if file_exists(model_path):
    print_skipped(
        "Step 5: Training seed detector",
        f"model already exists at {model_path}"
    )
else:
    if file_exists(train_detector_script):
        run_step(
            "Step 5: Training seed detector",
            ["python3", train_detector_script]
        )
    else:
        print_skipped(
            "Step 5: Training seed detector",
            f"{train_detector_script} not found"
        )

# ==================================================
# STEP 6: IMAGE PREDICTION
# ==================================================
predict_image_script = "src/object_detection/predict_seed_image.py"

if file_exists(predict_image_script):
    run_step(
        "Step 6: Predicting seed on images",
        ["python3", predict_image_script]
    )
else:
    print_skipped(
        "Step 6: Predicting seed on images",
        f"{predict_image_script} not found"
    )

# ==================================================
# STEP 7: VIDEO DETECTION
# ==================================================
detect_video_script = "src/object_detection/detect_video.py"
predict_video_script = "src/object_detection/predict_seed_video.py"

if file_exists(detect_video_script):
    run_step(
        "Step 7: Detecting seed on video",
        ["python3", detect_video_script]
    )
elif file_exists(predict_video_script):
    run_step(
        "Step 7: Detecting seed on video",
        ["python3", predict_video_script]
    )
else:
    print_skipped(
        "Step 7: Detecting seed on video",
        "no video detection script found"
    )

print("\nPipeline complete.")