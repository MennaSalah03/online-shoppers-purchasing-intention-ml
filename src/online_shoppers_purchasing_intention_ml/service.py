import structlog
from fastapi import FastAPI
from pydantic import BaseModel

from online_shoppers_purchasing_intention_ml.predict import IntentionPredictionModel

app = FastAPI()


class Entry(BaseModel):
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