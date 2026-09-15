# NYC Taxi Demand Prediction

This project is an end-to-end machine learning project for predicting the number of taxi pickups in a NYC pickup zone for a given hour.

I built this project mainly to learn how the different parts of an ML project fit together after training a model. Instead of stopping at model training, I wanted to have a pipeline that handles the data, trains and tracks models, serves predictions through an API, and runs tests automatically.

## What the project does

The model takes a pickup zone and some historical demand features and predicts the taxi demand for that location and hour.

The main features are:

* Pickup location
* Hour of the day
* Day of the week
* Demand from the previous hour
* Demand from the previous day
* Demand from the previous week
* 24-hour rolling demand
* 3-hour rolling demand

I used NYC Yellow Taxi trip data from January to May 2026. The data is aggregated by pickup location and hour before the features are created.

## Machine Learning

I trained a few models and compared them with a simple previous-hour baseline. The main models used in the final pipeline are:

* Random Forest
* XGBoost

Optuna was used for hyperparameter tuning.

The data was split chronologically rather than randomly since this is a time-based prediction problem. January through March are used for training and May is used for validation. June is used for testing. Addtionally, A future month can then be kept completely separate for final testing.

The Random Forest model improved substantially over the simple previous-hour baseline.

## MLOps

I used DVC to manage the datasets and the ML pipeline. The preprocessing and model training stages are defined in `dvc.yaml`, so the whole pipeline can be reproduced with DVC.

MLflow is used to track experiments, parameters and metrics, and to register trained models.

The project uses DAGsHub as the remote DVC storage.

The pipeline roughly looks like this:

```text
Raw taxi data
      ↓
DVC preprocessing pipeline
      ↓
Processed features
      ↓
Model training
      ↓
Random Forest / XGBoost
      ↓
MLflow experiment tracking
      ↓
Model artifacts
```

## API

The trained models are served using FastAPI.

There are two prediction endpoints:

```text
POST /predict/rf
POST /predict/xgb
```

Example request:

```json
{
    "PULocationID": 100,
    "hour": 7,
    "day_of_week": 3,
    "previous_hour": 20,
    "previous_day": 80,
    "previous_week": 300,
    "rolling_24h": 78,
    "rolling_3h": 40
}
```

The API returns the predicted taxi demand.

## Docker

The API is packaged into a Docker container so that the same environment can be used outside my local machine.

The container includes the FastAPI application and trained model files and exposes the API on port 8000.

## Testing and CI

I used pytest to test the FastAPI endpoints and input handling.

GitHub Actions runs the test suite whenever changes are pushed to the repository or a pull request is opened.

The CI workflow also pulls the required DVC pickel dumps before running the tests.

## Project structure

```text
.
├── .github/
│   └── workflows/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── process_data.py
│   ├── train.py
│   └── server.py
├── tests/
├── dvc.yaml
├── dvc.lock
├── Dockerfile
├── requirements.txt
└── pytest.ini
```

## Running the project

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Pull the DVC files:

```bash
dvc pull
```

To reproduce the pipeline:

```bash
dvc repro
```

To run the API locally:

```bash
uvicorn src.server:app --reload
```

The API will then be available at:

```text
http://localhost:8000
```

To run the tests:

```bash
pytest
```

To build the Docker image:

```bash
docker build -t nyc-taxi-api .
```

and run it with:

```bash
docker run -p 8000:8000 nyc-taxi-api
```

## Why I built this

The main goal of this project was to get more comfortable with the parts of machine learning that happen after model training.

I wanted to understand how data versioning, reproducible pipelines, experiment tracking, model serving, containers and automated testing fit together in one project.

There are still things that could be added to make the project more production-like, but I wanted to keep the scope reasonable rather than adding tools just for the sake of adding them.
