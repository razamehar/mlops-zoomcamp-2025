import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

experiment = client.get_experiment_by_name("random-forest-hyperopt")
if experiment is None:
    print("Experiment 'random-forest-hyperopt' not found.")
else:
    runs = client.search_runs(experiment.experiment_id)
    for run in runs:
        rmse = run.data.metrics.get("rmse")
        params = run.data.params
        print(f"Run ID: {run.info.run_id}, RMSE: {rmse}")
