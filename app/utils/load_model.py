import joblib

def load_model(model_path: str):
    return joblib.load(model_path)
