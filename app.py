from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os

app = FastAPI()

# Load the pre-trained model
MODEL_FILE = "my_model_one.h5"
model = load_model(MODEL_FILE)

detector = MTCNN()
TARGET_SIZE = (300, 300)

def preprocess_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_faces(frame)
        if results:
            x, y, w, h = results[0]['box']
            face = frame[y:y+h, x:x+w]

            if len(face.shape) == 2:
                face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)

            face = cv2.resize(face, TARGET_SIZE)
            frames.append(face)

    cap.release()
    return np.array(frames)

@app.post("/predict/")
async def predict_video(file: UploadFile = File(...)):
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    frames = preprocess_video(video_path)
    if frames.size == 0:
        return JSONResponse(content={"error": "No faces detected."})

    predictions = model.predict(frames)
    real_frames = np.sum(predictions[:, 1] > 0.5)
    fake_frames = np.sum(predictions[:, 0] > 0.5)

    video_category = "REAL" if real_frames > fake_frames else "FAKE"

    os.remove(video_path)  # Clean up the temp file

    return JSONResponse(content={"prediction": video_category})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
