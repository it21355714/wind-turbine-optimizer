import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.exception import CustomeException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomeException(e, sys)
    

def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            para = param.get(model_name, {})  # safer access, returns {} if not found


            gs=GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)



            report[list(models.keys())[i]]=test_model_score

        return report
    
    
    except Exception as e:
        raise CustomeException(e,sys)
    
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return {
        "R2": round(r2, 4),
        "MSE": round(mse, 4),
        "MAE": round(mae, 4)
    }, y_pred
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomeException(e, sys)

def save_plot(y_true, y_pred, title, path):
    plt.figure(figsize=(8, 5))
    plt.plot(y_true.values, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="orange")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
