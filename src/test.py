import mlflow

model = mlflow.get_model_version(
    name="random_forest_taxi_demand",
    version="2"
)

print("Source:", model.source)