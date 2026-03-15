# Timelapse Plant Phenotyping — Seed Detection

Small project to build a seed detection model from timelapse plant videos/frames using a Faster R-CNN detector (PyTorch / torchvision).

This repository contains scripts to extract frames, prepare a detection dataset (YOLO-style label text files), train a Faster R-CNN seed detector, and run detection on images or videos.

## Repository layout

- `data/` — dataset folders used by the scripts
  - `detection/images/{train,val}/<day_x>/` — image files
  - `detection/labels/{train,val}/<day_x>/` — per-image label `.txt` files (YOLO format)
  - `frames/` — extracted frames used by other scripts
  - `raw_videos/` — original mp4 videos
- `src/` — training, dataset, and utility code
  - `src/object_detection/detection_dataset.py` — dataset class (reads YOLO-like .txt labels and converts to boxes)
  - `src/object_detection/train_seed_detector.py` — training script (Faster R-CNN)
  - `src/object_detection/detect_image.py` / `detect_video.py` — inference helpers
- `models/` — where model checkpoints are saved (`seed_detector_fasterrcnn.pth` by default)
- `output/` — example outputs (detections, videos, etc.)

## Label format

Each label file is a text file with one object per line using the YOLO normalized format:

class_id x_center y_center width height

All values are floats. `class_id` should be `0` for seed (the training pipeline maps the foreground label to `1` in targets). Coordinates are normalized with respect to image width/height (range 0..1).

Example line:

0 0.5123 0.4234 0.05 0.06

If a label file is missing for an image the dataset loader will print a warning and skip that image.

## Quick start — train

1. Install dependencies. This project uses PyTorch + torchvision. On macOS with MPS support (Apple Silicon) you may want a matching PyTorch wheel. Example (adjust CUDA/MPS as appropriate):

```bash
# create and activate a python virtualenv (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# install PyTorch and torchvision (example for macOS with MPS; change for your platform)
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision
```

2. Verify your dataset is in `data/detection/images/train`, `data/detection/labels/train`, and equivalent `val` folders.

3. Run training (from project root):

```bash
python3 src/object_detection/train_seed_detector.py
```

Default training hyperparameters are defined at the top of `train_seed_detector.py` (batch size, lr, number of epochs). The script will print device (CUDA / MPS / CPU) and dataset sizes.

## Troubleshooting: exploding loss / NaN during training

If you see the loss explode and become `nan` (as in the training run output), check the following in order:

1. Label numeric ranges
   - The dataset expects YOLO-normalized numbers (0..1). If any label file contains non-normalized pixel coordinates or values outside [0,1], box coordinates will be wrong and the model loss may explode.
   - Inspect a few `.txt` files in `data/detection/labels/...` and ensure values look like floats between 0 and 1.

2. Empty / malformed lines
   - `detection_dataset.py` skips lines that don't have 5 parts. If some label files contain invalid lines you may silently drop annotations. Make sure each line follows `class x y w h`.

3. Degenerate boxes
   - Very small or zero-area boxes (width or height near zero) can cause unstable training. Ensure width and height are > 0 and reasonable.

4. Learning rate and pretrained weights
   - Try lowering the learning rate (e.g. 1e-3 -> 5e-4 or smaller) and/or use pretrained weights instead of training everything from scratch. In `train_seed_detector.py`, the model is created without pretrained weights by default.

5. Batch size / optimizer
   - Reduce batch size to 1 or use Adam for more stability while debugging:

```python
# example change in script
optimizer = torch.optim.Adam(params, lr=1e-4)
```

6. Gradient clipping
   - Add gradient clipping before optimizer.step() to avoid exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

7. Visual check
   - Visualize a few images with the converted boxes (convert normalized label -> pixel coords) to ensure boxes are where you expect.

If none of the above help, try running a single batch and print targets and loss components to find which part is NaN.

## Next steps / improvements

- Add a `requirements.txt` to pin package versions.
- Optionally use torchvision pretrained weights for the backbone to speed up convergence.
- Add small unit tests that validate label parsing for a few sample files.

## Contacts / license

This is a small personal project. Use, modify, and extend as you like.

---
Created on March 2026
