import pickle
from abc import ABC, abstractmethod

import structlog

from online_shoppers_purchasing_intention_ml.config import config

logger = structlog.get_logger()

class BasePredictionModel(ABC):
    @abstractmethod
    def load_model(self, model_path: str | None = None) -> "BasePredictionModel": ...

    @abstractmethod
    def predict_one(self, entry: dict) -> float: ...

    @abstractmethod
    def batch_predict(self, entries: list[dict]) -> list[float]: ...


class IntentionPredictionModel(BasePredictionModel):
    def __init__(self) -> None:
        self._dv = None
        self._model = None
        self._model_type = None
        self._auc = None
        self._threshold = 0.5
        self.logger = logger.bind(component="IntentionPredictionModel")

    def load_model(self, model_path = None) -> "IntentionPredictionModel":
        model_path = model_path or config.best_random_forest_model
        with open(model_path, "rb") as f_in:
            model_artifact = pickle.load(f_in)

            self._dv = model_artifact["dv"]
            self._model = model_artifact["model"]
            self._model_type = model_artifact["model_type"]
            self._auc = model_artifact["auc"]
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
        self._ensure_loaded()
        if not entries:
            return []
        cleaned_entries = [self._preprocess_entry(e) for e in entries]
        X = self._dv.transform(cleaned_entries)
        y_pred = self._model.predict_proba(X)[:, 1]
        predictions = []
        for p in y_pred:
            predictions.append({
                "purchase_probability": round(float(p), 4),
                "prediction": bool(float(p) >= self._threshold),
                "threshold": self._threshold,
                "auc": round(self._auc, 4)
            })
        return predictions

    def predict_one(self, entry: dict) -> float:
        return self.batch_predict([entry])[0]