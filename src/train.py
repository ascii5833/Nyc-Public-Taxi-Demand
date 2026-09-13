import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from utils import setupmlflow
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import optuna
import logging
import joblib
from pathlib import Path
from xgboost import XGBRegressor
import argparse

#paths
ROOT_DIR = Path(__file__).resolve().parent.parent
#configs
df_path = ROOT_DIR / "data" / "processed" /"taxi_demand_features_combined.parquet"
features = ["PULocationID", "hour", "day_of_week", "previous_hour", "previous_day", "previous_week",
        "rolling_24h", "rolling_3h"]


MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

#mlflow params
db_path = ROOT_DIR / "mlflow" / "mlflow.db"
db_path.parent.mkdir(parents = True, exist_ok = True)
uri  = f"sqlite:///{db_path.as_posix()}"
# uri = "sqlite:\\..\\mlflow\\mlflow.db"
experiment_name = "nyc_taxi_demand"

#optuna params
N_TRIALS = 100

#setup loggers
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def _rfObj(trial, X_train, y_train, X_val, y_val) -> int:
    '''
    Optuna objective to maximize cross validation r^2 square.
    
    '''
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"])
    }
    model = RandomForestRegressor(**params, n_jobs = -1, random_state = 42)
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    
    return r2


    

#random forest mlflow
def tuneRF(data_split : list[tuple[pd.DataFrame]] = None):
    X_train, X_val, X_test = data_split[0][0], data_split[1][0], data_split[2][0]
    y_train, y_val, y_test = data_split[0][1], data_split[1][1], data_split[2][1]
    
    #set optuna study
    study = optuna.create_study(
        study_name = "random_forest_tuning",
        direction = "maximize",
        sampler = optuna.samplers.TPESampler(n_startup_trials = 20, n_ei_candidates = 24 , multivariate = True, seed = 42, group = True)
    )
    
    study.optimize(
        lambda trial : _rfObj(trial, X_train, y_train, X_val, y_val),
        n_trials = N_TRIALS,
        show_progress_bar = True
        
    )
    
    best = study.best_params
    
    log.info(f"Random Forest best params {best} CV (R2) = {study.best_value}")
    
    
    #model to store
    model = RandomForestRegressor(**best, n_jobs = -1, random_state = 42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    #mlflow run
    with mlflow.start_run(run_name = "RF_OP_Best") as run:
        mlflow.log_params(best)
        mlflow.log_param("random_state", 42)
        mlflow.log_metrics({
            "mae": mae,
            "rmse":rmse,
            "r2": r2,
            "n_trials": N_TRIALS
        })
        #log the model
        model_info = mlflow.sklearn.log_model(
            sk_model = model,
            name = "random_forest_taxi_demand"
        )
        run_id = run.info.run_id
    
    joblib.dump(model, MODEL_DIR / "tuned_RF.pkl")
    
    return {"model" : model_info,  "name": "tuned_rf", "r2" : r2, "rmse" : rmse, "mae" : mae, "best_params" : best, "run_id" : run_id}

#xgbobjective
def _xgbObj(trial, X_train, y_train, X_val, y_val) -> int:
    '''
    Optuna xgboost objective to maximize cross validation r^2 square.
    
    '''
    params = {
        "n_estimators": 5000,
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "tree_method": "hist",
        "booster": "gbtree",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "random_state":42,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "early_stopping_rounds" : 50,
        "reg_alpha" : trial.suggest_float("reg_alpha", 1e-3, 10.0, log = True),
        "reg_lambda" : trial.suggest_float("reg_lambda", 1.0, 50.0, log = True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20)
    }
    
    model = XGBRegressor(**params, n_jobs = -1,)
    model.fit(X_train, y_train, eval_set = [(X_val,y_val)], verbose = False)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    
    return r2

def tuneXGB(data_split : list[tuple[pd.DataFrame]] = None):
    X_train, X_val, X_test = data_split[0][0], data_split[1][0], data_split[2][0]
    y_train, y_val, y_test = data_split[0][1], data_split[1][1], data_split[2][1]
    
    #set optuna study
    study = optuna.create_study(
        study_name = "xgb_tuning",
        direction = "maximize",
        sampler = optuna.samplers.TPESampler(n_startup_trials = 20, n_ei_candidates = 24 , multivariate = True, seed = 42, group = True)
    )
    
    study.optimize(
        lambda trial : _xgbObj(trial, X_train, y_train, X_val, y_val),
        n_trials = N_TRIALS,
        show_progress_bar = True
        
    )
    
    best = study.best_params
    final_params = {
        "n_estimators" : 5000,
        "tree_method" : "hist",
        "booster" : "gbtree",
        "random_state": 42,
        "early_stopping_rounds" : 50,
        **best
    }
    log.info(f"XGBOOST best params {best} CV (R2) = {study.best_value}")
    
    
    #model to store
    model = XGBRegressor(**final_params, n_jobs = -1)
    model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = False)
    y_pred = model.predict(X_test)
    #metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    #mlflow run
    with mlflow.start_run(run_name = "xgb_OP_Best") as run:
        mlflow.log_params(best)
        mlflow.log_param("random_state", 42)
        mlflow.log_metrics({
            "mae": mae,
            "rmse":rmse,
            "r2": r2,
            "n_trials": N_TRIALS
        })
        #log the model
        model_info = mlflow.xgboost.log_model(
            xgb_model = model,
            name = "xgboost_taxi_demand",
        )
        run_id = run.info.run_id
    
    joblib.dump(model, MODEL_DIR / "tuned_XGB.pkl")
    
    return {"model" : model_info,  "name": "tuned_XGB", "r2" : r2, "rmse" : rmse, "mae" : mae, "best_params" : best, "run_id" : run_id}  
        
#splitter
def split_data(df : pd.DataFrame, time_splits : list[str]) -> tuple[pd.DataFrame]:
    train_df = df[df['pickup_hour'] < time_splits[0]].copy() 
    val_df = df[df['pickup_hour'].between(time_splits[0], time_splits[1], inclusive='left')].copy()
    test_df = df[df['pickup_hour'].between(time_splits[1], time_splits[2], inclusive='left')].copy()
    
    return (train_df, val_df, test_df)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Script to run the ml models")
    parser.add_argument("-m", "--model", type = str, help = "model type")
    
    args = parser.parse_args()
    model = "rf"
    
    if args.model:
        assert args.model in ["xgb", "rf"], "Incorrect model specified as argument"
        model = args.model 
    
    
            
    df = pd.read_parquet(df_path)
    #splits
    time_splits = ['2026-04-01', '2026-05-01', '2026-06-01']
    train_df, val_df, test_df = split_data(df, time_splits)

    X_train = train_df[features]
    y_train = train_df["number_pickups"]
    
    X_val = val_df[features]
    y_val = val_df["number_pickups"]
    
    X_test = test_df[features]
    y_test = test_df["number_pickups"]
    
    #setup mlflow
    setupmlflow(uri, experiment_name)
    res = ""
    if model == "rf":
        #random forest optimization
        rf_res = tuneRF([(X_train, y_train), (X_val, y_val), (X_test, y_test)])
        
        #register best model
        registered_rf = mlflow.register_model(
            model_uri = f"models:/{rf_res['model'].model_id}",
            name = "random_forest_taxi_demand"
        )
        res = rf_res
    elif model == "xgb":
        #xgboost optimization
        xgb_res = tuneXGB([(X_train, y_train), (X_val, y_val), (X_test, y_test)])
        
        #register best model
        registered_rf = mlflow.register_model(
            model_uri = f"models:/{xgb_res['model'].model_id}",
            name = "xgboost_taxi_demand"
        )
        res = xgb_res
       
    print(res)
    
    
    
    

    
    

    
    