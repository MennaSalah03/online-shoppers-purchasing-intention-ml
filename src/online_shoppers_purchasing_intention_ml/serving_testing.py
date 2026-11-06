import json

import requests
import structlog

from online_shoppers_purchasing_intention_ml.config import config
from online_shoppers_purchasing_intention_ml.train import load_data, split_data

logger = structlog.get_logger()


def test_health_endpoint():
    response = requests.get(f"{config.api_url}/health", timeout=5)
    response.raise_for_status()
    logger.info("health_check_passed", response=response.json())


def test_raw_single_prediction():
    """Tests /predict using a hardcoded raw dictionary (simulating an external client)."""
    raw_entry = {
        "administrative": 3.0,
        "administrative_duration": 87.5,
        "informational": 0.0,
        "informational_duration": 0.0,
        "productrelated": 27.0,
        "productrelated_duration": 798.33,
        "bouncerates": 0.0,
        "exitrates": 0.0126,
        "pagevalues": 22.91,
        "specialday": 0.8,
        "month": "Feb",
        "operatingsystems": 2,
        "browser": 2,
        "region": 3,
        "traffictype": 1,
        "visitortype": "Returning_Visitor",
        "weekend": 0
    }

    response = requests.post(f"{config.api_url}/predict", json=raw_entry, timeout=5)
    response.raise_for_status()
    logger.info("raw_single_prediction_result", result=response.json())


def test_endpoints_from_dataset():
    logger.info("loading_test_data", path=config.data_path)
    df = load_data(config.data_path)
    _, X_test, _, y_test = split_data(df)

    single_record = json.loads(X_test.iloc[0].to_json())
    
    batch_records = json.loads(X_test.iloc[100:110].to_json(orient="records"))
    batch_actual_labels = y_test[100:110].tolist()

    logger.info("testing_single_predict")
    resp_single = requests.post(f"{config.api_url}/predict", json=single_record)
    resp_single.raise_for_status()
    
    logger.info("single_predict_success", response=resp_single.json())

    logger.info("testing_batch_predict", batch_size=len(batch_records))
    resp_batch = requests.post(f"{config.api_url}/predict/batch", json=batch_records)
    resp_batch.raise_for_status()
    
    logger.info(
        "batch_predict_success", 
        predictions=resp_batch.json(), 
        actual_labels=batch_actual_labels
    )
if __name__ == "__main__":
    # /health endpoint test
    test_health_endpoint()

    # /predict test with raw data
    test_raw_single_prediction()

    # /predict and /predict batch testing from test dataset
    test_endpoints_from_dataset()