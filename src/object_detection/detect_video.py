import os
import cv2
import torch
import torchvision.ops as ops
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
 
# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = "models/seed_detector_fasterrcnn.pth"
INPUT_FOLDER = "data/raw_videos"
OUTPUT_FOLDER = "output/detection_videos"
 
NUM_CLASSES = 3
FRAME_SKIP = 1
NMS_IOU_THRESHOLD = 0.5
 
CLASS_NAMES = {
    1: "seed",
    2: "germination"
}
 
# Per-class confidence thresholds
CLASS_THRESHOLDS = {
    1: 0.30,   # seed
    2: 0.25,   # germination — slightly lower, harder to detect early
}
 
# Per-class bounding box colors (BGR)
CLASS_COLORS = {
    1: (0, 255, 0),      # Green  — seed
    2: (0, 165, 255),    # Orange — germination
}
# ──────────────────────────────────────────────────────────────────────────────
 
 
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
 
 
def apply_nms_per_class(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """
    Apply Non-Maximum Suppression independently for each class so that
    duplicate overlapping boxes on the same object are removed, while
    still allowing a seed box and a germination box to coexist.
    """
    if not detections:
        return []
 
    final = []
    class_ids = set(d["label"] for d in detections)
 
    for cls_id in class_ids:
        cls_dets = [d for d in detections if d["label"] == cls_id]
        if len(cls_dets) == 1:
            final.extend(cls_dets)
            continue
 
        boxes_t = torch.tensor([d["box"] for d in cls_dets], dtype=torch.float32)
        scores_t = torch.tensor([d["score"] for d in cls_dets], dtype=torch.float32)
        keep = ops.nms(boxes_t, scores_t, iou_threshold)
        final.extend([cls_dets[i] for i in keep.tolist()])
 
    return final
 
 
def draw_detections(frame, detections, stale=False):
    """
    Draw bounding boxes and labels for all detections on the frame.
    Stale detections (carried forward from the last frame) are drawn
    with thinner boxes and a '(last)' suffix so you can tell them apart.
    """
    for det in detections:
        xmin, ymin, xmax, ymax = map(int, det["box"])
        label = det["label"]
        score = det["score"]
        class_name = CLASS_NAMES.get(label, f"class_{label}")
        color = CLASS_COLORS.get(label, (255, 255, 255))
        thickness = 1 if stale else 2
 
        # Bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
 
        # Label text
        suffix = " (last)" if stale else ""
        label_text = f"{class_name}: {score:.2f}{suffix}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
 
        label_y = max(th + 6, ymin - 6)
 
        # Filled background rectangle for readability
        cv2.rectangle(
            frame,
            (xmin, label_y - th - 4),
            (xmin + tw + 4, label_y + baseline),
            color,
            -1
        )
        # Black text on top
        cv2.putText(
            frame,
            label_text,
            (xmin + 2, label_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness
        )
 
 
def process_video(model, device, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open video: {input_path}")
        return
 
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10
 
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
    frame_count = 0
    current_detections = []   # detections for the current processed frame
    last_detections    = []   # carried forward when current frame has none
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        # ── Run inference every FRAME_SKIP frames ─────────────────────────
        if frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            image_tensor = F.to_tensor(pil_img).to(device)
 
            with torch.no_grad():
                outputs = model([image_tensor])
 
            output = outputs[0]
            boxes  = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
 
            # Collect every detection that clears its per-class threshold
            raw_detections = []
            for i in range(len(scores)):
                predicted_label = int(labels[i])
                threshold = CLASS_THRESHOLDS.get(predicted_label, 0.3)
                if scores[i] >= threshold:
                    raw_detections.append({
                        "box":   boxes[i],
                        "score": float(scores[i]),
                        "label": predicted_label,
                    })
 
            # NMS per class to remove duplicate boxes
            current_detections = apply_nms_per_class(raw_detections)
 
            # Keep the last good set for temporal smoothing
            if current_detections:
                last_detections = current_detections
 
        # ── Draw detections ───────────────────────────────────────────────
        if current_detections:
            draw_detections(frame, current_detections, stale=False)
        elif last_detections:
            # No detection this frame — carry forward the last known boxes
            draw_detections(frame, last_detections, stale=True)
 
        out.write(frame)
        frame_count += 1
 
        if frame_count % 30 == 0:
            seed_count  = sum(1 for d in current_detections if d["label"] == 1)
            germ_count  = sum(1 for d in current_detections if d["label"] == 2)
            print(
                f"  {os.path.basename(input_path)}: "
                f"frame {frame_count:>5} | "
                f"seeds: {seed_count} | "
                f"germinations: {germ_count}"
            )
 
    cap.release()
    out.release()
    print(f"  Saved: {output_path}  ({frame_count} frames total)")
 
 
def main():
    # ── Validate paths ────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Run train_seed_detector.py first.")
        return
 
    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder not found: {INPUT_FOLDER}")
        return
 
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
 
    # ── Load model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.\n")
 
    # ── Process videos ────────────────────────────────────────────────────
    videos = sorted(
        f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".mp4")
    )
 
    if not videos:
        print(f"No .mp4 files found in {INPUT_FOLDER}")
        return
 
    print(f"Found {len(videos)} video(s): {videos}\n")
 
    for video_file in videos:
        input_path  = os.path.join(INPUT_FOLDER, video_file)
        output_name = video_file.replace(".mp4", "_seed_detected.mp4")
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        print(f"Processing: {video_file}")
        process_video(model, device, input_path, output_path)
        print()
 
 
if __name__ == "__main__":
    main()