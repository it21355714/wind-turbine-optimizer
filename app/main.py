from fastapi import FastAPI, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List
import numpy as np
from .schemas import PredictionInput, PredictionOutput, PredictionSchema
from .database import create_table, get_db, save_prediction, delete_prediction, get_all_predictions
from .utils.load_model import load_model
from fastapi.middleware.cors import CORSMiddleware
import os

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the predictions table when the app starts
create_table()

# Load models and scalers using joblib
model_path = r'E:\4th year\research project version 2\model train\model_train_env\app\models\mlp_model_noice_optimization.pkl'
scaler_X_path = r'E:\4th year\research project version 2\model train\model_train_env\app\models\scaler_mlp_X_noice_optimization.pkl'
scaler_y_path = r'E:\4th year\research project version 2\model train\model_train_env\app\models\scaler_mlp_y_noice_optimization.pkl'

model = load_model(model_path)
scaler_X = load_model(scaler_X_path)
scaler_y = load_model(scaler_y_path)

@app.post("/predict/", response_model=PredictionOutput)
def predict(data: PredictionInput, db: Session = Depends(get_db), save: bool = Query(False)):
    """
    Predict the rotor speed, power output, and optimized pitch angle based on the wind speed and direction.
    Optionally, save the prediction to the database.
    """
    try:
        wind_speed = data.wind_speed
        wind_dir = data.wind_direction
        desired_noise = data.noise_level  # Could be -1 to skip optimization

        if desired_noise == -1:
            default_pitch = -1.91  # Default pitch as in your script
            input_data = np.array([[wind_speed, wind_dir, default_pitch]])
            input_data_scaled = scaler_X.transform(input_data)
            prediction_scaled = model.predict(input_data_scaled)
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(1, -1)).flatten()

            rotor_speed, power_output, optimized_pitch_angle = prediction[1], prediction[0], default_pitch
        else:
            best_pitch = None
            min_diff = float('inf')
            best_prediction = None

            for pitch in np.arange(-3, 10, 0.1):
                input_data = np.array([[wind_speed, wind_dir, pitch]])
                input_data_scaled = scaler_X.transform(input_data)
                prediction_scaled = model.predict(input_data_scaled)
                prediction = scaler_y.inverse_transform(prediction_scaled.reshape(1, -1)).flatten()

                noise = prediction[2]
                diff = abs(noise - desired_noise)

                if diff < min_diff:
                    min_diff = diff
                    best_pitch = pitch
                    best_prediction = prediction

            rotor_speed, power_output, optimized_pitch_angle = best_prediction[1], best_prediction[0], best_pitch

        if save:
            save_prediction(wind_speed, wind_dir, desired_noise, optimized_pitch_angle, power_output, rotor_speed)
            print("Prediction successfully saved!")

        return PredictionOutput(
            rotor_speed=rotor_speed,
            power_output=power_output,
            optimized_pitch_angle=optimized_pitch_angle,
            noise_level=desired_noise
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.delete("/delete/{prediction_id}")
def delete_saved_prediction(prediction_id: int):
    result = delete_prediction(prediction_id)
    if not result:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"message": f"Prediction ID {prediction_id} deleted successfully"}

@app.get("/predictions/", response_model=List[PredictionSchema])
def read_predictions(db: Session = Depends(get_db)):
    """
    Get all saved predictions from the database.
    """
    rows = get_all_predictions(db)

    return [
        PredictionSchema(
            id=row[0],
            wind_speed=row[1],
            wind_direction=row[2],
            noise_level=row[3],
            optimized_pitch_angle=row[4],
            power_output=row[5],
            rotor_speed=row[6],
            timestamp=row[7]
        )
        for row in rows
    ]
