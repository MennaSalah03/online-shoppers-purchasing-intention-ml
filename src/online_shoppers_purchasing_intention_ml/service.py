from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from pydantic import BaseModel

from online_shoppers_purchasing_intention_ml.predict import IntentionPredictionModel

logger = structlog.get_logger()

class Entry(BaseModel):
    administrative: float
    administrative_duration: float
    informational: float
    informational_duration: float
    productrelated: float
    productrelated_duration: float
    bouncerates: float
    exitrates: float
    pagevalues: float
    specialday: float
    month: str
    operatingsystems: int
    browser: int
    region: int
    traffictype: int
    visitortype: str
    weekend: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    # loading model
    try:
        predictor.load_model()
        predictor.logger.info("api_startup_complete", msg="Model loaded successfully.")
    except Exception as e:
        predictor.logger.exception("api_startup_failed", error=str(e))
    yield
    # shutdown api
    predictor.logger.info("api_shutdown_complete")


app = FastAPI(title="Online Shoppers Intention API", lifespan=lifespan)

predictor = IntentionPredictionModel()


@app.post("/predict")
async def predict_one(entry: dict):
    result = predictor.predict_one(entry)
    return result

@app.post("/predict/batch")
async def predict_batch(entries: list[dict]):
    dict_entries = [e for e in entries]
    results = predictor.batch_predict(dict_entries)
    return {"predictions": results}

@app.get("/health", description="retreive health info about the loaded model")
async def health():
    is_loaded = (predictor._model is not None) and (predictor._dv is not None)
    return {
        "status": "healthy" if is_loaded else "Model not found",
        "model_type": predictor._model_type if is_loaded else None,
    }
