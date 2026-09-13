import mlflow
import yaml

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


def load_data_version(dv_lck_path : str):
    with open(dv_lck_path) as f:
        dvc_lock = yaml.safe_load(f)
    
    dvc_hash = ""
      
    for output in dvc_lock["stages"]["preprocess"]["outs"]:
        dvc_hash = output["md5"]
        break
    
    return dvc_hash
        