from threading import Lock
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="YOLOv8 Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_model_lock = Lock()
_model: Any | None = None
_model_load_error: str | None = None

BACKEND_DIR = Path(__file__).resolve().parent.parent
CUSTOM_MODEL_PATH = BACKEND_DIR / "runs" / "shelf_detector" / "weights" / "best.pt"
DEFAULT_MODEL = "yolov8n.pt"


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
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


def _nms(
    boxes: list[tuple[int, int, int, int, float]], iou_threshold: float = 0.35
) -> list[tuple[int, int, int, int, float]]:
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list[tuple[int, int, int, int, float]] = []

    for candidate in sorted_boxes:
        cbox = (candidate[0], candidate[1], candidate[2], candidate[3])
        if all(_iou(cbox, (k[0], k[1], k[2], k[3])) < iou_threshold for k in kept):
            kept.append(candidate)
    return kept


def _split_large_box(
    gray: Any,
    x: int,
    y: int,
    w: int,
    h: int,
) -> list[tuple[int, int, int, int, float]]:
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

    separators: list[int] = [0]
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

    candidates: list[tuple[int, int, int, int, float]] = []
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


def fallback_detect_objects(image_bgr: Any, conf: float) -> list[dict[str, Any]]:
    """Detect generic object-like regions using OpenCV contours when YOLO is unavailable."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Fallback detector unavailable: {exc}") from exc

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

    raw_boxes: list[tuple[int, int, int, int, float]] = []

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
        relaxed_boxes: list[tuple[int, int, int, int, float]] = []
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

    detections: list[dict[str, Any]] = []
    for x1, y1, x2, y2, score in final_boxes:
        detections.append(
            {
                "classId": -1,
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
            }
        )

    return detections


def get_model() -> Any:
    global _model
    global _model_load_error
    if _model_load_error is not None:
        raise RuntimeError(_model_load_error)

    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from ultralytics import YOLO

                    model_path = CUSTOM_MODEL_PATH if CUSTOM_MODEL_PATH.exists() else DEFAULT_MODEL
                    _model = YOLO(str(model_path))
                except Exception as exc:  # noqa: BLE001
                    _model_load_error = str(exc)
                    raise RuntimeError(_model_load_error) from exc
    return _model


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect")
async def detect_objects(
    image: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.01, le=1.0),
) -> dict[str, Any]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    file_bytes = await image.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"OpenCV runtime unavailable: {exc}") from exc

    np_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        model = get_model()
        inference = model.predict(source=image_rgb, conf=conf, verbose=False)

        if not inference:
            return {
                "fileName": image.filename,
                "width": width,
                "height": height,
                "model": "yolov8",
                "detections": [],
            }

        result = inference[0]
        detections: list[dict[str, Any]] = []
        class_names = result.names

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "classId": cls_id,
                        "className": class_names.get(cls_id, str(cls_id)),
                        "confidence": round(confidence, 4),
                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                            "width": round(x2 - x1, 2),
                            "height": round(y2 - y1, 2),
                        },
                    }
                )

        return {
            "fileName": image.filename,
            "width": width,
            "height": height,
            "model": "yolov8",
            "detections": detections,
        }
    except RuntimeError as yolo_error:
        try:
            fallback_detections = fallback_detect_objects(image_bgr=image_bgr, conf=conf)
        except RuntimeError as fallback_error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "YOLOv8 and fallback detection are unavailable on this machine. "
                    f"YOLO error: {yolo_error}. Fallback error: {fallback_error}."
                ),
            ) from fallback_error

        return {
            "fileName": image.filename,
            "width": width,
            "height": height,
            "model": "opencv-fallback",
            "warning": f"YOLOv8 unavailable on this machine: {yolo_error}",
            "detections": fallback_detections,
        }
