# YOLOv8 Full-Stack Assignment Starter

This workspace now contains:

- `backend/`: FastAPI API for image object detection using YOLOv8.
- `frontend/`: React + Vite UI to upload images and visualize detections.
- `sample_images/`: extracted sample images from your zip file.

## 1) Backend Setup (FastAPI + YOLOv8)

Open a terminal in `backend/` and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API will be available at:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## 2) Frontend Setup (React + Vite)

Open another terminal in `frontend/` and run:

```powershell
npm install
npm run dev
```

Frontend will run on `http://127.0.0.1:5173`.

## 3) How Detection Works

1. UI sends image to `POST /detect`.
2. Backend loads YOLOv8 (`yolov8n.pt`) and predicts objects.
3. API returns bounding boxes, class names, and confidence.
4. Frontend overlays boxes on the uploaded image.

## 4) Notes

- First YOLOv8 run may take longer because model weights are downloaded.
- If you want higher accuracy, change model in `backend/app/main.py` from `yolov8n.pt` to `yolov8s.pt` or `yolov8m.pt`.
- Keep backend running while using frontend.
