import mlflow

def setupmlflow(uri : str, experiment_name : str = "mlflow_experiment"):
    #mlflow setup
    mlflow.set_tracking_uri(
        uri
    )
    
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Experiment:", mlflow.get_experiment_by_name(experiment_name))
    
