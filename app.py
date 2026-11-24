import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Inicializar la aplicación FastAPI
app = FastAPI(
    title="Microservicio de Inferencia Iris",
    description="API REST contenerizada para predicción de especies de la flor Iris.",
    version="1.0.0",
)

# 2. Cargar el artefacto del modelo al iniciar la API
try:
    artifact = joblib.load("iris_model.pkl")
    model = artifact["model"]
    target_names = artifact["target_names"]
except Exception as e:
    raise RuntimeError(f"Error al cargar el archivo del modelo: {e}")


# 3. Esquema de validación estricta para la petición con Pydantic
class IrisRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Longitud del sépalo en cm")
    sepal_width: float = Field(..., example=3.5, description="Ancho del sépalo en cm")
    petal_length: float = Field(..., example=1.4, description="Longitud del pétalo en cm")
    petal_width: float = Field(..., example=0.2, description="Ancho del pétalo en cm")


# 4. Endpoint de comprobación de estado (Health Check)
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Iris Classifier Microservice",
        "dataset_info": "150 instancias, 4 características, 3 clases",
    }


# 5. Endpoint de inferencia POST
@app.post("/predict")
def predict_species(request: IrisRequest):
    try:
        # Convertir las características a formato vectorial (1, 4)
        features = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]])

        # Realizar la predicción y obtener probabilidades
        prediction_id = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        predicted_class = target_names[prediction_id]
        confidence = float(probabilities[prediction_id])

        return {
            "prediction": predicted_class,
            "class_id": prediction_id,
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(prob), 4)
                for name, prob in zip(target_names, probabilities)
            },
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error en la inferencia: {str(err)}")
