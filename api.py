import os
import gdown
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

# =====================
#   1) Model Download
# =====================

MODEL_PATH = "model.pt"
DRIVE_URL = "https://drive.google.com/uc?id=1xJkhGgcUvCmcMd3SDhL5xhDVMF1T5Khd"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully.")
else:
    print("Model already exists. Skipping download.")

# =====================
#   2) Load YOLO model
# =====================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# =====================
#   3) FastAPI Setup
# =====================

API_KEY = "mysecret"   # غيّرها لمفتاحك

app = FastAPI()

# Allow all origins — عشان تقدر تشغله في الفرونت
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
#   4) Health Check
# =====================
@app.get("/")
def home():
    return {"status": "running", "model": MODEL_PATH}

# =====================
#   5) Prediction API
# =====================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):

    # Check API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API KEY")

    # Read image
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # YOLO inference
    results = model.predict(img, conf=0.25)
    r = results[0]

    boxes = []
    if r.boxes is not None:
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            boxes.append({
                "xyxy": xyxy,
                "conf": conf,
                "class": cls
            })

    return {
        "num_boxes": len(boxes),
        "boxes": boxes
    }


# Start server (local only)
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
