from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#root
ROOT_DIR = Path(__file__).resolve().parent.parent
#models_path
MODEL_DIR = ROOT_DIR / "models"
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

@app.post("/predict/{model_name}")
async def get_prediction(model_name :str,data : TaxiDemand):
    try:
         
        if model_name not in models:
            if model_name == "rf":
                model_path = MODEL_DIR / "tuned_RF.pkl"
            
            elif model_name == "xgb":
                model_path = MODEL_DIR / "tuned_XGB.pkl"
                
            else:
                raise HTTPException( status_code=400, detail=f"Unknown model: {model_name}" )
            
            models[model_name] = joblib.load(model_path)
                
            logger.info(f"{model_name} loaded")
        
        input_df = pd.DataFrame([data.model_dump()])
        prediction = models[model_name].predict(input_df)
        
        return {"Prediction": prediction.tolist()}
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code = 500, detail = str(e))
        



    