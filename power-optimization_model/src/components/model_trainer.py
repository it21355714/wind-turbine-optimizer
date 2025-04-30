# model_trainer.py
import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src.exception import CustomeException
from src.logger import logging
from src.utils import save_object, evaluate_models, evaluate_model, save_plot

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost Classifier": XGBRegressor(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50],
                    'max_features': ['sqrt'],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1],
                    'n_estimators': [50],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1],
                    'n_estimators': [50],
                }
            }

            model_report = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            best_model_score = max(sorted(model_report.values()))
            best_models_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_models_name]

            if best_model_score < 0.6:
                raise CustomeException("no best model found")

            logging.info("Best found model on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomeException(e, sys)

# Utility functions for additional tasks
def train_energy_forecasting_models():
    from src.components.data_transformation import get_features_and_target

    df=pd.read_csv('notebook\data\wind_data1.csv', sep=',', encoding="utf-8", engine="python")
    if 'Date and time' in df.columns:
     df = df.drop(columns=['Timestamp'])
    X, y = get_features_and_target(df, target_col='Power (kW)')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(),
        "XGBoost Classifier": XGBRegressor(),
    }

    metrics = {}
    graph_paths = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        scores, y_pred = evaluate_model(model, X_test, y_test)
        metrics[name] = scores

        graph_path = f"static/energy_{name.lower().replace(' ', '_')}.png"
        save_plot(y_test, y_pred, f"Energy Forecasting - {name}", graph_path)
        graph_paths.append(graph_path)

    return metrics, graph_paths

def train_power_curve_models():
    from src.components.data_transformation import get_features_and_target

    df=pd.read_csv('notebook\data\wind_data1.csv', sep=',', encoding="utf-8", engine="python")
    if 'Date and time' in df.columns:
        df = df.drop(columns=['Timestamp'])
    X, y = get_features_and_target(df, target_col='Power')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(),
    }

    metrics = {}
    graph_paths = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        scores, y_pred = evaluate_model(model, X_test, y_test)
        metrics[name] = scores

        graph_path = f"static/power_curve_{name.lower().replace(' ', '_')}.png"
        save_plot(y_test, y_pred, f"Power Curve Analysis - {name}", graph_path)
        graph_paths.append(graph_path)

    return metrics, graph_paths
