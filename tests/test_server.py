from src.server import app
from fastapi.testclient import TestClient

client = TestClient(app)

#sending this package to the server
payload = { 
           "PULocationID": 100, 
           "hour": 7, 
           "day_of_week": 3,
           "previous_hour": 20, 
           "previous_day": 80, 
           "previous_week": 300, 
           "rolling_24h": 78, 
           "rolling_3h": 40 }


def test_root():
    response = client.get("/")
    
    assert response.status_code == 200


def test_xgb():
    '''
    status code =  200,
    result has Prediction,
    result is a list
    result list is of size 1
    '''
    
    response = client.post("/predict/xgb", json = payload)
    
    assert response.status_code == 200
    
    result = response.json()
    
    assert "Prediction" in result
    
    assert isinstance(result['Prediction'], list)
    
    assert len(result['Prediction']) == 1
    
    
def test_rf():
    '''
    status code =  200,
    result has Prediction,
    result is a list
    result list is of size 1
    '''
    
    response = client.post("/predict/rf", json = payload)
    
    assert response.status_code == 200
    
    result = response.json()
    
    assert "Prediction" in result
    
    assert isinstance(result['Prediction'], list)
    
    assert len(result['Prediction']) == 1
    
    
