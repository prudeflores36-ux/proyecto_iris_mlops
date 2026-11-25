from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
from pydantic import BaseModel, Field
import joblib
import numpy as np

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

model = joblib.load("iris_model.pkl")

app = FastAPI(
    title="Iris Classifier Microservice",
    version="1.0.0",
    description="API REST con FastAPI y scikit-learn para clasificar especies de Iris."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0.0, le=20.0)
    sepal_width: float = Field(..., ge=0.0, le=20.0)
    petal_length: float = Field(..., ge=0.0, le=20.0)
    petal_width: float = Field(..., ge=0.0, le=20.0)

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "iris-classifier-api",
        "endpoints": ["GET /", "GET /ui", "POST /predict"]
    }

@app.get("/ui", response_class=HTMLResponse)
def ui():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")

@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]], dtype=float)

    class_id = int(model.predict(features)[0])
    prediction = CLASS_NAMES[class_id]

    if hasattr(model, "predict_proba"):
        probs_raw = model.predict_proba(features)[0]
        probabilities = {
            CLASS_NAMES[i]: round(float(probs_raw[i]), 4)
            for i in range(len(CLASS_NAMES))
        }
        confidence = round(float(np.max(probs_raw)), 4)
    else:
        probabilities = {name: (1.0 if i == class_id else 0.0) for i, name in enumerate(CLASS_NAMES)}
        confidence = 1.0

    return {
        "prediction": prediction,
        "class_id": class_id,
        "confidence": confidence,
        "probabilities": probabilities
    }
