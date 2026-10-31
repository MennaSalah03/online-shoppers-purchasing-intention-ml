import pickle
from abc import ABC, abstractmethod

import structlog

from online_shoppers_purchasing_intention_ml.config import config

logger = structlog.get_logger()

class BasePredictionModel(ABC):
    @abstractmethod
    def load_model(self, model_path: str | None = None) -> "BasePredictionModel": ...

    @abstractmethod
    def predict(self, entry: dict) -> float: ...

    @abstractmethod
    def batch_predict(self, entries: list[dict]) -> list[float]: ...


class IntentionPredictionModel(BasePredictionModel):
    def __init__(self) -> None:
        self._dv = None
        self._model = None
        self.logger = logger.bind(component="IntentionPredictionModel")

    def load_model(self, model_path = None) -> "IntentionPredictionModel":
        model_path = model_path or config.model_path
        with open(model_path, "rb") as f_in:
            # model_artifact consisting of a tuple (dv, model)
            model_artifact = pickle.load(f_in)

            self._dv, self._model = model_artifact
            logger.info("model_artifact_loaded_successfully")
        return self

    def _ensure_loaded(self) -> None:
        if self._dv is None or self._model is None:
            raise RuntimeError("Model artifact is not loaded. Call `load_model()` before predicting.")
    @staticmethod
    def _preprocess_entry(entry: dict) -> dict:
        cleaned = entry.copy()
        for col in config.categorical:
            if col in cleaned:
                cleaned[col] = str(cleaned[col])
        for col in config.numerical:
            if col in cleaned:
                cleaned[col] = float(cleaned[col])
        return cleaned
    def batch_predict(self, entries: list[dict]) -> list[float]:
        predictions = []
        for entry in entries:
            predictions.append(self.predict(entry))

        return predictions


        def predict_one(self, entry: dict) -> float:
            pass        
