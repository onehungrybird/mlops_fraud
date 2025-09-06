# from mlflow.tracking import MlflowClient
# client = MlflowClient(tracking_uri="http://98.80.224.211:5000")
# print(client.get_registered_model("credit-card-fraud-model"))



import mlflow.pyfunc
mlflow.set_tracking_uri("http://98.80.224.211:5000")
model_path = "models:/credit-card-fraud-model/Staging"
model = mlflow.pyfunc.load_model(model_path)
if model:
    print('model loaded')
else:
    print("model not loaded")