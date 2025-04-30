from flask import Flask,request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app=application
##Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            wind_speed=float(request.form.get('wind_speed')),
            Rotor_speed=float(request.form.get('Rotor_speed')),
            Nacelle_position=float(request.form.get('Nacelle_position')),
            Blade_angle=float(request.form.get('Blade_angle'))
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=round(results[0], 2))
    
@app.route('/energy-forecasting', methods=['GET', 'POST'])
def energy_forecasting():
    if request.method == 'GET':
        return render_template('energy_forecasting.html')
    else:
        # Get user input from form
        input_data = CustomData(
            wind_speed=float(request.form.get('wind_speed')),
            Rotor_speed=float(request.form.get('Rotor_speed')),
            Nacelle_position=float(request.form.get('Nacelle_position')),
            Blade_angle=float(request.form.get('Blade_angle'))
        )
        df = input_data.get_data_as_data_frame()

        # Load model and make prediction
        forecasting_model = EnergyForecastingModel()
        result = forecasting_model.predict(df)

        return render_template('energy_forecasting.html', results=round(result[0], 2))

@app.route('/power-curve-analysis')
def power_curve_analysis():
    from src.components.model_trainer import train_power_curve_models
    metrics, graph_paths = train_power_curve_models()
    return render_template('power_curve_analysis.html', metrics=metrics, graphs=graph_paths)


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)