from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import requests
import pandas as pd

app = Flask(__name__)

# Load models
classifier = joblib.load('models/XGBclassifier_model.pkl')
yaw_model = joblib.load('models/xgb_yaw_model.pkl')
pitch_model = joblib.load('models/xgb_pitch_model.pkl')
rotor_model = joblib.load('models/xgb_rotor_model.pkl')
xgb_weather_model = joblib.load('models/xgb_weather_risk_model.pkl') 

# Define your feature orderhttp://127.0.0.1:5000/predict
FEATURE_ORDER = [
    'ActivePower', 'AirPress1', 'EGDInRotorSpd', 'EGDYawPositionToNorth',
    'Hum1', 'PitchBlade1', 'PitchBlade2', 'Precipitation',
    'Temp1', 'WD187m', 'Windspeed87m'
]

FEATURE_WEATHER = ['AirPress1', 'Hum1', 'Precipitation', 'Temp1', 'WD187m', 'Windspeed87m']

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_ORDER)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        input_values = [float(data.get(feat, 0)) for feat in FEATURE_ORDER]
        input_array = np.array(input_values).reshape(1, -1)
        risk_prediction = classifier.predict(input_array)[0]

        if risk_prediction == 1:
            weather_values = [float(data.get(feat, 0)) for feat in FEATURE_WEATHER]
            weather_array = np.array(weather_values).reshape(1, -1)
            yaw = float(yaw_model.predict(weather_array)[0])
            pitch = float(pitch_model.predict(weather_array)[0])
            rotor_speed = float(rotor_model.predict(weather_array)[0])

            result = {
                "Risk": "⚠️ Risky Condition Detected",
                "Suggested Yaw": f"{yaw:.2f}",
                "Suggested Pitch": f"{pitch:.2f}",
                "Suggested Rotor Speed": f"{rotor_speed:.2f}"
            }
        else:
            result = {
                "Risk": "✅ No risk detected",
                "Turbine Status": "Operating Normally"
            }

        return render_template('index.html', features=FEATURE_ORDER, result=result)

    except Exception as e:
        return render_template('index.html', features=FEATURE_ORDER, result={"Error": str(e)})

@app.route('/forecast-risk', methods=['GET'])
def forecast_risk():
    try:
        API_key = "cf29f2a6f1d42dcdba9d62fc1670bbc0"
        lat = 39.912083
        lon = -105.220056
        URL = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_key}&units=metric"

        response = requests.get(URL)
        data = response.json()

        forecast_data = []
        for entry in data["list"]:
            dt = entry["dt_txt"]
            main = entry["main"]
            wind = entry["wind"]
            rain = entry.get("rain", {})

            row = {
                "datetime": dt,
                "AirPress1": main["pressure"],
                "Hum1": main["humidity"],
                "Temp1": main["temp"],
                "Precipitation": rain.get("3h", 0),
                "WD187m": wind["deg"],
                "Windspeed87m": wind["speed"]
            }
            forecast_data.append(row)

        forecast_df = pd.DataFrame(forecast_data)
        features = FEATURE_WEATHER
        X_forecast = forecast_df[features]
        forecast_df['Predicted_Risk'] = xgb_weather_model.predict(X_forecast)

        risky_slots = forecast_df[forecast_df['Predicted_Risk'] == 1]
        if risky_slots.empty:
            return jsonify({"message": "✅ No risky weather conditions detected in the upcoming forecast."})

        suggestion_input = risky_slots[features]
        risky_slots['Suggested_Yaw'] = yaw_model.predict(suggestion_input)
        risky_slots['Suggested_Pitch'] = pitch_model.predict(suggestion_input)
        risky_slots['Suggested_RotorSpd'] = rotor_model.predict(suggestion_input)

        result = risky_slots[['datetime', 'Windspeed87m', 'WD187m', 'Temp1',
                              'Suggested_Yaw', 'Suggested_Pitch', 'Suggested_RotorSpd']].to_dict(orient='records')

        return jsonify({
            "message": "⚠️ Risky weather conditions forecasted. Suggested actions below.",
            "recommendations": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
