import pickle

import pandas as pd
import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from online_shoppers_purchasing_intention_ml.config import config

logger = structlog.get_logger()

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    for col in config.categorical:
        if col in df.columns:
            df[col] = df[col].astype(str)
                
    for col in config.numerical:
        if col in df.columns:
            df[col] = df[col].astype(float)
                
    return df

def split_data(df: pd.DataFrame):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=config.seed)

    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train[config.target].values
    y_test = df_test[config.target].values

    del df_full_train[config.target]
    del df_test[config.target]

    return df_full_train, df_test, y_full_train, y_test

def train_model(df_train: pd.DataFrame, y_train):
    dv = DictVectorizer(sparse=False)
    
    features = config.categorical + config.numerical
    train_dicts = df_train[features].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    model = RandomForestClassifier(
        n_estimators=config.n,
        max_depth=config.d,
        min_samples_leaf=config.s,
        random_state=config.seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return dv, model

def save_model_artifact(dv, model, output_path: str):
    artifact = {
        "dv": dv,
        "model": model,
        "model_type": type(model).__name__
    }
    with open(output_path, "wb") as f_out:
        pickle.dump(artifact, f_out)


if __name__ == "__main__":
    logger.info("loading_and_preparing_data", data_path=config.data_path)
    df = load_data(config.data_path)
    
    logger.info("splitting_data", split_ratio="80/20")
    df_full_train, _, y_full_train, _ = split_data(df)
    
    logger.info("training_final_model_on_full_train")
    dv, final_model = train_model(df_full_train, y_full_train)
    
    logger.info("saving_model_artifact", output_path=config.best_random_forest_model)
    save_model_artifact(dv, final_model, config.best_random_forest_model
    )
    
    logger.info("training_script_complete")