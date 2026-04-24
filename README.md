# Shelf Product Detection & Grouping AI Pipeline

This project provides an end-to-end AI pipeline for detecting retail shelf items and grouping them, exposed via a FastAPI backend and a React frontend.

## 🚀 Project Overview

The AI pipeline is designed to:
1. Accept image uploads via a minimal frontend UI.
2. Send images to a FastAPI-based backend.
3. Detect products on retail shelves using YOLOv8 (with a fallback OpenCV-based heuristic detector).
4. Group products (simulated via ID assignment).
5. Return structured JSON data containing coordinates and groups.
6. Visualize the output via the frontend.

## 🏗️ Architecture Design & Data Flow

1. **Client (React UI)**: Sends `multipart/form-data` image to the backend.
2. **Backend (FastAPI)**: Receives the image, validates it, and converts it to a NumPy array.
3. **Detector Microservice (YOLOv8)**:
   - Evaluates the image using pre-trained `yolov8n.pt`.
   - If YOLO is unavailable (e.g., due to system security policies or missing weights), the system gracefully falls back to an **OpenCV Edge-Density Heuristic Detector**.
4. **Grouping Algorithm**: Groups detected objects and assigns unique brand/group IDs.
5. **Response**: A formatted JSON containing bounding boxes and grouping IDs is returned to the client and visualized.

## 🛠️ Setup & Installation Instructions

### 1. Backend Setup (FastAPI + YOLOv8)

Open a terminal in the `backend/` folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- `http://127.0.0.1:8000/docs` (Swagger UI for testing)

### 2. Frontend Setup (React + Vite)

Open a new terminal in the `frontend/` folder and run:

```powershell
npm install
npm run dev
```

The UI will run on `http://localhost:5173`. Upload an image there to see the bounding boxes.

---

## 🤖 Auto-Training YOLOv8 (Custom Shelf Detection)

Standard pre-trained YOLOv8 detects generic objects (e.g., cups, bottles). Since the provided sample dataset lacks manual annotations, I have built an automated self-supervised learning pipeline. 

You can auto-generate a dataset and train YOLOv8 on shelf items using the included script:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python auto_train.py
```

**What this script does:**
1. Iterates over the raw images in `sample_images/`.
2. Uses our OpenCV fallback heuristic to generate bounding boxes (pseudo-labels).
3. Formats them into standard YOLO annotation `.txt` files.
4. Generates a `shelf.yaml` data config.
5. Initiates YOLOv8 transfer learning (`model.train()`).

*Note: You may need to run this on a machine without strict Application Control Policies, as Windows might block PyTorch/Pydantic DLL execution locally.*

---

## 📄 Input/Output JSON Formats

### 1. Input Format (Client -> Server)

The application accepts an image via a standard HTTP POST request.
- **Endpoint:** `POST /detect`
- **Headers:** `Content-Type: multipart/form-data`
- **Body:** `image` file payload. Optionally includes a `conf` (confidence threshold) query parameter.

### 2. Output Format (Server -> Client)

The backend returns a standardized JSON format. This makes it highly scalable and easy for any client to parse and visualize.

```json
{
  "fileName": "shelf_image.jpg",
  "width": 1920,
  "height": 1080,
  "model": "yolov8",
  "detections": [
    {
      "classId": 0,
      "className": "product",
      "groupId": 1,
      "confidence": 0.8942,
      "bbox": {
        "x1": 150.5,
        "y1": 200.0,
        "x2": 250.0,
        "y2": 350.5,
        "width": 99.5,
        "height": 150.5
      }
    }
  ]
}
```

## 🧠 Alternative Approaches & Comparisons

1. **YOLOv8 + OpenCV Fallback (Current Approach):** 
   - *Pros:* Extremely fast inference, lightweight, falls back gracefully on edge devices without GPUs.
   - *Cons:* OpenCV heuristics might struggle with heavily occluded items or extreme lighting variations compared to pure Deep Learning models.

2. **Zero-Shot Object Detection (e.g., GroundingDINO):**
   - *Approach:* We could prompt a model with "retail shelf products" to detect items without training.
   - *Pros:* Requires absolutely zero training or annotations.
   - *Cons:* Extremely high latency (can take several seconds per image) and high compute requirements, breaking the "minimal latency" requirement of the assignment.

3. **Pre-trained SKU-110K Model:**
   - *Approach:* Download a model explicitly trained on the dense SKU-110K dataset.
   - *Pros:* Highly accurate for densely packed shelves out-of-the-box.
   - *Cons:* Heavy reliance on large external weight files, less adaptable if the product definition changes.

**Conclusion:** Our approach balances minimal latency, robustness (via the fallback engine), and a clear scalable architecture using microservices, perfectly fulfilling the assignment constraints.
