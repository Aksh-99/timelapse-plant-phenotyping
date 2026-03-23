import os
import cv2
import torch
import torchvision.ops as ops
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
 
# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/seed_detector_fasterrcnn.pth"
INPUT_FOLDER  = "data/raw_videos"
OUTPUT_FOLDER = "output/detection_videos"
 
NUM_CLASSES       = 6      # background + seed + germination
FRAME_SKIP        = 1
NMS_IOU_THRESHOLD = 0.5
DEBUG_FRAMES      = 10     # print raw scores for first N frames per video
 
# Per-class confidence thresholds (lowered for better recall)
CLASS_THRESHOLDS = {
    1: 0.10,   # seed
    2: 0.08,   # germination
    3: 0.08,   # sprout
    4: 0.08,   # seedling
    5: 0.08,   # vegetative
}
 
CLASS_NAMES = {
    1: "seed",
    2: "germination",
    3: "sprout",
    4: "seedling",
    5: "vegetative"
}
 
# Per-class bounding box colors (BGR)
CLASS_COLORS = {
    1: (0, 0, 255),      # seed (RED)
    2: (0, 165, 255),    # germination (ORANGE)
    3: (0, 255, 255),    # sprout (YELLOW)
    4: (0, 255, 0),      # seedling (GREEN)
    5: (255, 0, 0),      # vegetative (BLUE)
}
# ──────────────────────────────────────────────────────────────────────────────
 
 
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
 
 
def verify_model_load(model_path, device):
    """Check checkpoint keys match the model and warn on mismatch."""
    model = get_model(NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
 
    model_keys       = set(model.state_dict().keys())
    checkpoint_keys  = set(checkpoint.keys())
    missing          = model_keys - checkpoint_keys
    unexpected       = checkpoint_keys - model_keys
 
    print("\n── Model Load Verification ──────────────────────")
    print(f"  Model keys     : {len(model_keys)}")
    print(f"  Checkpoint keys: {len(checkpoint_keys)}")
    if missing:
        print(f"  [WARN] Missing keys    ({len(missing)}): {list(missing)[:5]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}")
    if not missing and not unexpected:
        print("  [OK] Keys match perfectly.")
    print("─────────────────────────────────────────────────\n")
 
    model.load_state_dict(checkpoint)
    return model
 
 
def apply_nms_per_class(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """
    Apply Non-Maximum Suppression independently per class so duplicate
    boxes on the same object are removed, while seed and germination
    boxes can still coexist in the same frame.
    """
    if not detections:
        return []
 
    final = []
    for cls_id in set(d["label"] for d in detections):
        cls_dets = [d for d in detections if d["label"] == cls_id]
        if len(cls_dets) == 1:
            final.extend(cls_dets)
            continue
        boxes_t  = torch.tensor([d["box"]   for d in cls_dets], dtype=torch.float32)
        scores_t = torch.tensor([d["score"] for d in cls_dets], dtype=torch.float32)
        keep     = ops.nms(boxes_t, scores_t, iou_threshold)
        final.extend([cls_dets[i] for i in keep.tolist()])
 
    return final
 
 
def draw_detections(frame, detections, stale=False):
    """
    Draw all bounding boxes and labels on the frame.
    Stale = carried forward from the previous frame (no detection this frame).
    """
    for det in detections:
        xmin, ymin, xmax, ymax = map(int, det["box"])
        label      = det["label"]
        score      = det["score"]
        class_name = CLASS_NAMES.get(label, f"class_{label}")
        color      = CLASS_COLORS.get(label, (255, 255, 255))
        thickness  = 1 if stale else 2
 
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
 
        suffix     = " (last)" if stale else ""
        label_text = f"{class_name}: {score:.2f}{suffix}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thick = 2
 
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)
        label_y = max(th + 6, ymin - 6)
 
        # Filled label background for readability
        cv2.rectangle(
            frame,
            (xmin, label_y - th - 4),
            (xmin + tw + 4, label_y + baseline),
            color, -1
        )
        cv2.putText(
            frame, label_text, (xmin + 2, label_y),
            font, font_scale, (0, 0, 0), font_thick
        )
 
 
def debug_print_scores(frame_count, scores, labels, boxes):
    """Print raw model output to help diagnose detection problems."""
    print(f"\n  ── DEBUG Frame {frame_count} ──────────────────────────────")
    print(f"     Total raw detections : {len(scores)}")
    if len(scores) == 0:
        print("     [!] Model returned NO detections for this frame.")
        print("         Possible causes:")
        print("         1. NUM_CLASSES mismatch (check training config)")
        print("         2. Model weights did not load correctly")
        print("         3. Input frame is blank or corrupt")
    else:
        print(f"     Top-5 detections:")
        for i in range(min(5, len(scores))):
            cname = CLASS_NAMES.get(int(labels[i]), f"class_{int(labels[i])}")
            print(f"       [{i}] label={int(labels[i])} ({cname}), "
                  f"score={scores[i]:.4f}, box={[round(float(x), 1) for x in boxes[i]]}")
        unique_labels = sorted(set(int(l) for l in labels))
        print(f"     Unique labels in output : {unique_labels}")
        print(f"     Score range : min={scores.min():.4f}  max={scores.max():.4f}")
        if scores.max() < 0.10:
            print("     [!] All scores very low — model may not recognise these frames.")
            print("         Check NUM_CLASSES matches your training config.")
        elif scores.max() < 0.30:
            print("     [!] Scores below old 0.30 threshold but above new 0.10 —")
            print("         detections should now appear with updated thresholds.")
    print("  ──────────────────────────────────────────────────────")
 
 
def process_video(model, device, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open: {input_path}")
        return
 
    fps    = cap.get(cv2.CAP_PROP_FPS) or 10
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Resolution : {width}x{height}  |  FPS: {fps:.1f}  |  Frames: {total}")
 
    # Save a sample frame for visual inspection
    ret_check, sample_frame = cap.read()
    if ret_check:
        sample_path = output_path.replace(".mp4", "_sample_frame.jpg")
        cv2.imwrite(sample_path, sample_frame)
        print(f"  Sample frame saved → {sample_path}  (inspect to confirm video loads OK)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start
 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
    frame_count        = 0
    current_detections = []
    last_detections    = []
    total_seed_frames      = 0
    total_germ_frames      = 0
    total_sprout_frames    = 0
    total_seedling_frames  = 0
    total_vegetative_frames = 0
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        if frame_count % FRAME_SKIP == 0:
            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img      = Image.fromarray(rgb)
            image_tensor = F.to_tensor(pil_img).to(device)
 
            with torch.no_grad():
                outputs = model([image_tensor])
 
            output = outputs[0]
            boxes  = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
 
            # ── Debug output for first N frames ──────────────────────────
            if frame_count < DEBUG_FRAMES:
                debug_print_scores(frame_count, scores, labels, boxes)
 
            # ── Filter by per-class threshold ─────────────────────────────
            raw_detections = []
            for i in range(len(scores)):
                predicted_label = int(labels[i])
                threshold = CLASS_THRESHOLDS.get(predicted_label, 0.10)
                if scores[i] >= threshold:
                    raw_detections.append({
                        "box":   boxes[i],
                        "score": float(scores[i]),
                        "label": predicted_label,
                    })
 
            current_detections = apply_nms_per_class(raw_detections)
 
            if current_detections:
                last_detections    = current_detections
                total_seed_frames += any(d["label"] == 1 for d in current_detections)
                total_germ_frames += any(d["label"] == 2 for d in current_detections)
                total_sprout_frames += any(d["label"] == 3 for d in current_detections)
                total_seedling_frames += any(d["label"] == 4 for d in current_detections)
                total_vegetative_frames += any(d["label"] == 5 for d in current_detections)

        # ── Draw ──────────────────────────────────────────────────────────
        if current_detections:
            draw_detections(frame, current_detections, stale=False)
        elif last_detections:
            draw_detections(frame, last_detections, stale=True)
 
        out.write(frame)
        frame_count += 1
 
        if frame_count % 30 == 0:
            seed_n = sum(1 for d in current_detections if d["label"] == 1)
            germ_n = sum(1 for d in current_detections if d["label"] == 2)
            sprout_n = sum(1 for d in current_detections if d["label"] == 3)
            seedling_n = sum(1 for d in current_detections if d["label"] == 4)
            vegetative_n = sum(1 for d in current_detections if d["label"] == 5)
            print(f"  {os.path.basename(input_path)}: "
                  f"frame {frame_count:>5} | seeds: {seed_n} | germinations: {germ_n}")
 
    cap.release()
    out.release()
 
    print(f"\n  ── Summary: {os.path.basename(input_path)} ──────────────────")
    print(f"     Total frames        : {frame_count}")
    print(f"     Frames with seed    : {total_seed_frames}")
    print(f"     Frames with germin. : {total_germ_frames}")
    if total_seed_frames == 0 and total_germ_frames == 0:
        print("     [!] Zero detections across the entire video.")
        print("         → Set DEBUG_FRAMES=9999 to inspect all score outputs.")
        print("         → Verify NUM_CLASSES matches your training config.")
    print(f"     Saved → {output_path}")
    print("  ─────────────────────────────────────────────────────\n")
 
 
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Run train_seed_detector.py first.")
        return
 
    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder not found: {INPUT_FOLDER}")
        return
 
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    # Load with key verification
    model = verify_model_load(MODEL_PATH, device)
    model.to(device)
    model.eval()
    print("Model ready.\n")
 
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
 
 
if __name__ == "__main__":
    main()
 
