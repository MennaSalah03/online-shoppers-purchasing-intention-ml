
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
       "configs"
       api_url: str = "http://0.0.0.0:8000"
       
       seed: int = 42
       data_path: str = "data/online_shoppers_intention.csv"
       
       best_random_forest_model: str = "models/train_script_rf_d=10_s=10_n=40.pkl"
       n: int = 40
       d: int = 10
       s: int = 10
       

       categorical: list[str] = ['visitortype', 'month', 'browser', 'region', 'traffictype', 'weekend']
       numerical: list[str] = ['administrative', 'administrative_duration', 'informational',
              'informational_duration', 'productrelated', 'productrelated_duration',
              'bouncerates', 'exitrates', 'pagevalues', 'specialday', 'operatingsystems'
              ]
       target: str = "revenue"

config = Settings()