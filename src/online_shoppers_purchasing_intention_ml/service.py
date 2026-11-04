import structlog
from fastapi import FastAPI
from pydantic import BaseModel

from online_shoppers_purchasing_intention_ml.predict import IntentionPredictionModel

logger = structlog.get_logger()

app = FastAPI()


class Entry(BaseModel):
<<<<<<< HEAD
    administrative: float
    administrative_duration: float
    informational: float
    informational_duration: float
    productrelated:float
    productrelated_duration: float
    bouncerates: float
    exitrates: float
    pagevalues: float
    specialday: int
    month: str
    operatingsystems: int
    browser: int
    region: int
    traffictype: int
    visitortype: str
    weekend: int
    pass


def load_predictor():
    pass

@app.post("/predict")
async def predict(entry: dict):
    pass

@app.post("/predict/batch")
async def predict_batch(entries: list[dict]):
    pass

@app.get("/health", description="retreive health info about the loaded model")
async def health():
    pass