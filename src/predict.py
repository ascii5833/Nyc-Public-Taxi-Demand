from fastapi import FastAPI, HTTPException
import mlflow
from pydantic import BaseModel
from utils import setupmlflow
from pathlib import Path
import pandas as pd
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#root
ROOT_DIR = Path(__file__).resolve().parent.parent
db_path = ROOT_DIR / "mlflow" / "mlflow.db"
db_path.parent.mkdir(parents = True, exist_ok = True)

#setup mlflow
uri  = f"sqlite:///{db_path.as_posix()}"
experiment_name = "nyc_taxi_demand"


app = FastAPI()
# Model input template
class TaxiDemand(BaseModel):
    PULocationID : int
    hour : int
    day_of_week : int
    previous_hour : float
    previous_day : float
    previous_week : float
    rolling_24h : float
    rolling_3h : float
    
    
@app.get("/")
async def read_root():
    return {"This server predicts NYC Taxi demand when given input historical data."}

models = {}

@app.post("/predict/{model_name}/{model_version}")
async def get_prediction(model_name :str, model_version :str,data : TaxiDemand):
    try:
        model_uri = f"models:/{model_name}/{model_version}"

        if model_uri not in models:
            models[model_uri] = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"{model_uri} loaded")
        
        input_df = pd.DataFrame([data.model_dump()])
        prediction = models[model_uri].predict(input_df)
        
        return {"Prediction": prediction.tolist()}
        
    
    except Exception as e:
        raise HTTPException(status_code = 404, detail = str(e))
        


if __name__ == "__main__":
    #mlflow setup
    setupmlflow(uri, experiment_name)
    uvicorn.run("predict:app", host = "127.0.0.1", port = 8000, reload = True)
    