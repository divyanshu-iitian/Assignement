import os
import shutil
import cv2
import yaml
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom

def _nms(boxes, iou_threshold=0.35):
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    for candidate in sorted_boxes:
        cbox = (candidate[0], candidate[1], candidate[2], candidate[3])
        if all(_iou(cbox, (k[0], k[1], k[2], k[3])) < iou_threshold for k in kept):
            kept.append(candidate)
    return kept

def _split_large_box(gray, x, y, w, h):
    import cv2
    import numpy as np
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return []
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    roi_edges = cv2.Canny(roi_blur, 70, 180)
    vertical_profile = roi_edges.sum(axis=0).astype(np.float32)
    if vertical_profile.size < 20:
        return []
    kernel = np.ones(15, dtype=np.float32) / 15.0
    smooth = np.convolve(vertical_profile, kernel, mode="same")
    low_threshold = float(np.percentile(smooth, 35))
    low_mask = smooth <= low_threshold
    separators = [0]
    start = None
    for idx, is_low in enumerate(low_mask):
        if is_low and start is None:
            start = idx
        if not is_low and start is not None:
            if idx - start >= 5:
                separators.append((start + idx) // 2)
            start = None
    if start is not None and len(low_mask) - start >= 5:
        separators.append((start + len(low_mask)) // 2)
    separators.append(w)
    separators = sorted(set(max(0, min(w, s)) for s in separators))
    min_segment_width = max(24, int(w * 0.06))
    max_segment_width = max(min_segment_width + 1, int(w * 0.45))
    candidates = []
    for i in range(len(separators) - 1):
        sx1 = separators[i]
        sx2 = separators[i + 1]
        seg_w = sx2 - sx1
        if seg_w < min_segment_width or seg_w > max_segment_width:
            continue
        seg = roi_edges[:, sx1:sx2]
        density = float(seg.mean()) / 255.0
        if density < 0.035:
            continue
        score = min(0.92, 0.35 + density * 1.6)
        candidates.append((x + sx1, y, x + sx2, y + h, score))
    return candidates

def fallback_detect_objects(image_bgr, conf):
    import cv2
    import numpy as np
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 170)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = width * height
    min_area_ratio = 0.0008 + (conf * 0.006)
    min_area = max(250, int(image_area * min_area_ratio))
    max_area = int(image_area * 0.30)
    raw_boxes = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:120]:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 18 or h < 18:
            continue
        if w > int(width * 0.82) and h < int(height * 0.45):
            continue
        box_ratio = w / max(h, 1)
        if box_ratio > 5.8 or box_ratio < 0.12:
            continue
        score = min(0.95, 0.30 + (area / image_area) * 3.5)
        if w > int(width * 0.28) and h > int(height * 0.18):
            split_candidates = _split_large_box(gray=gray_eq, x=x, y=y, w=w, h=h)
            if len(split_candidates) >= 3:
                raw_boxes.extend(split_candidates)
                continue
        raw_boxes.append((x, y, x + w, y + h, float(score)))
    final_boxes = _nms(raw_boxes, iou_threshold=0.33)[:60]
    if not final_boxes:
        relaxed_boxes = []
        relaxed_min_area = max(120, int(image_area * 0.00035))
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:140]:
            area = cv2.contourArea(contour)
            if area < relaxed_min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 14 or h < 14:
                continue
            if w > int(width * 0.90) and h < int(height * 0.55):
                continue
            score = min(0.9, 0.26 + (area / image_area) * 3.0)
            relaxed_boxes.append((x, y, x + w, y + h, float(score)))
        final_boxes = _nms(relaxed_boxes, iou_threshold=0.28)[:80]
    detections = []
    for x1, y1, x2, y2, score in final_boxes:
        detections.append({
            "classId": 0,
            "className": "object",
            "confidence": round(float(score), 4),
            "bbox": {
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "width": round(float(x2 - x1), 2),
                "height": round(float(y2 - y1), 2),
            },
        })
    return detections

def create_dataset():
    print("Creating dataset from sample images using OpenCV fallback detector...")
    
    # Paths
    base_dir = BASE_DIR
    src_images_dir = base_dir / "sample_images" / "sample_images"
    
    dataset_dir = base_dir / "backend" / "dataset"
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    
    # Create directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Process each image
    for img_name in os.listdir(src_images_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = src_images_dir / img_name
        image_bgr = cv2.imread(str(img_path))
        
        if image_bgr is None:
            continue
            
        height, width = image_bgr.shape[:2]
        
        # Use fallback detector to get pseudo-labels
        # conf=0.25 to get a reasonable amount of bounding boxes
        detections = fallback_detect_objects(image_bgr, conf=0.25)
        
        # Copy image to dataset directory
        dst_img_path = images_dir / img_name
        shutil.copy(img_path, dst_img_path)
        
        # Create YOLO format label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = labels_dir / label_name
        
        with open(label_path, "w") as f:
            for det in detections:
                box = det["bbox"]
                # YOLO format: class_id center_x center_y width height (normalized)
                x_center = (box["x1"] + box["x2"]) / 2.0 / width
                y_center = (box["y1"] + box["y2"]) / 2.0 / height
                box_width = box["width"] / width
                box_height = box["height"] / height
                
                # Class 0: product
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                
    print(f"Dataset created successfully at {dataset_dir}")
    
    # Create YAML file
    yaml_path = base_dir / "backend" / "shelf.yaml"
    yaml_content = {
        "path": str(dataset_dir.absolute()),
        "train": "images/train",
        "val": "images/train",  # Using train for val just for this demo
        "names": {0: "product"}
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"Dataset configuration saved to {yaml_path}")
    return yaml_path

def train_yolo(yaml_path):
    print("Starting YOLOv8 training...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Please run: pip install ultralytics")
        return
        
    # Load a pre-trained model (starting with nano for speed)
    model = YOLO("yolov8n.pt")
    
    # Train the model
    # We use 10 epochs for demonstration purposes so it doesn't take forever
    results = model.train(
        data=str(yaml_path),
        epochs=10,
        imgsz=640,
        project=str(BACKEND_DIR / "runs"),
        name="shelf_detector",
        exist_ok=True
    )
    print("Training complete! New weights saved in backend/runs/shelf_detector/weights/best.pt")

if __name__ == "__main__":
    yaml_path = create_dataset()
    train_yolo(yaml_path)
