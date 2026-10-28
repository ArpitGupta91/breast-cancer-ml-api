
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"

model = joblib.load(MODEL_PATH)
FEATURE_NAMES = list(model.feature_names_in_)

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Educational ML model prediction service",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {
        "message": "Breast Cancer Classification API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(request: PredictionRequest):
    supplied = request.features

    missing = set(FEATURE_NAMES) - set(supplied)
    extra = set(supplied) - set(FEATURE_NAMES)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": sorted(missing),
                "unexpected_features": sorted(extra)
            }
        )

    try:
        row = pd.DataFrame(
            [[supplied[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES
        )

        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]

        return {
            "prediction": (
                "malignant" if prediction == 1 else "benign"
            ),
            "class_id": prediction,
            "probabilities": {
                str(int(label)): float(probability)
                for label, probability
                in zip(model.classes_, probabilities)
            },
            "note": "Educational prediction only; not a medical diagnosis."
        }

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values: {exc}"
        )
