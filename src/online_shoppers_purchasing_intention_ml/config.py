
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
       "configs"
       seed: int = 42
       data_path: str = "data/online_shoppers_intention.csv"
       model_path = "models/shopper_intention_model_C=1.bin"

       # data colums
       categorical: list[str] = ['visitortype', 'month']
       numerical: list[str] = ['administrative', 'administrative_duration', 'informational',
              'informational_duration', 'productrelated', 'productrelated_duration',
              'bouncerates', 'exitrates', 'pagevalues', 'specialday',
              'operatingsystems', 'browser', 'region', 'traffictype', 'weekend'
       ]
       target: str = "revenue"

config = Settings()