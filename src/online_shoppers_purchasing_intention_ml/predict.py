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

            self._dv = model_artifact["dv"]
            self._model = model_artifact["model"]
        return self
