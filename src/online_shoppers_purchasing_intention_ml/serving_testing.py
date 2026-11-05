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
        "weekend": 0,
    }

    response = requests.post(f"{config.api_url}/predict", json=raw_entry, timeout=5)
    response.raise_for_status()
    logger.info("raw_single_prediction_result", result=response.json())



if __name__ == "__main__":
    # /health endpoint test
    test_health_endpoint()

    # /predict and /predict batch testing from test dataset
    df = load_data(config.data_path)
    _, X_test, _, y_test = split_data(df)

    # the test entries
    entry = X_test.iloc[0].to_json(indent=2)
    record = json.loads(entry)
    
    records = []
    for i in range(100, 110):
        records.append(json.loads(X_test.iloc[i].to_json(indent=2)))

    resp_single = requests.post(f"{config.api_url}/predict", json=record)
    print("Single:", resp_single.json())

    resp_batch = requests.post(f"{config.api_url}/predict/batch", json=records)
    print("Batch:", resp_batch.json())
    print(y_test[100:110])
