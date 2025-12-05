from fastapi import FastAPI, UploadFile
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model = YOLO("epoch37.pt")

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    results = model.predict(img)
    return {"detections": results[0].boxes.xyxy.tolist(),
            "classes": results[0].boxes.cls.tolist(),
            "conf": results[0].boxes.conf.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
