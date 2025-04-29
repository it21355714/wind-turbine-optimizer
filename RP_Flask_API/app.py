from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import sqlite3
import os

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Load the trained LSTM model
MODEL_PATH = "model/SLIIT_lstm_wind_turbine_model.keras"  # Ensure correct path
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Initialize SQLite Database
DB_PATH = "database.db"

# ✅ Create SQLite database if not exists
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rpm REAL,
                torque REAL,
                power_out REAL,
                t1 REAL,
                t2 REAL,
                t3 REAL,
                event_count REAL,
                windspeed_ref REAL,
                voltage_L1 REAL,
                voltage_L2 REAL,
                current_out REAL,
                event_status REAL,
                turbine_status REAL,
                failure_detected TEXT,
                rotor_status TEXT,
                gearbox_status TEXT,
                generator_status TEXT,
                temperature_status TEXT,
                event_log_status TEXT,
                overall_status TEXT
            )
        ''')
        conn.commit()

init_db()  # Initialize database on app start

# ✅ Serve the frontend HTML page
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        required_fields = ["rpm", "torque", "power_out", "t1", "t2", "t3", 
                           "event_count", "windspeed_ref", "voltage_L1", 
                           "voltage_L2", "current_out", "event_status", "turbine_status"]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        input_data = np.array([[ 
            data["rpm"], data["torque"], data["power_out"], 
            data["t1"], data["t2"], data["t3"], 
            data["event_count"], data["windspeed_ref"],
            data["voltage_L1"], data["voltage_L2"], 
            data["current_out"], data["event_status"], 
            data["turbine_status"]
        ]], dtype=np.float32)

        input_data = input_data.reshape(1, 1, 13)

        prediction = model.predict(input_data)
        prediction_class = int(prediction[0][0] > 0.5)

        rotor_status = "Normal" if data["rpm"] < 3200 else "High RPM - Potential Rotor Imbalance"
        gearbox_status = "Normal" if data["torque"] > 20 else "Possible Gearbox Issue"
        generator_status = "Normal" if data["power_out"] > 500 else "Low Power - Generator Issue"
        temperature_status = "Normal" if all(t < 50 for t in [data["t1"], data["t2"], data["t3"]]) else "High Temperature - Cooling System Issue"
        event_log_status = "Normal" if data["event_status"] == 0 else "Event Error Detected"

        overall_status = "Failure Detected" if prediction_class == 1 else "Operating Normally"

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (rpm, torque, power_out, t1, t2, t3, event_count, windspeed_ref,
                                        voltage_L1, voltage_L2, current_out, event_status, turbine_status,
                                        failure_detected, rotor_status, gearbox_status, generator_status, 
                                        temperature_status, event_log_status, overall_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data["rpm"], data["torque"], data["power_out"], 
                data["t1"], data["t2"], data["t3"], 
                data["event_count"], data["windspeed_ref"],
                data["voltage_L1"], data["voltage_L2"], 
                data["current_out"], data["event_status"], 
                data["turbine_status"], 
                "Yes" if prediction_class == 1 else "No",
                rotor_status, gearbox_status, generator_status,
                temperature_status, event_log_status, overall_status
            ))
            conn.commit()

        return jsonify({
            "message": "Prediction successful",
            "input": data,
            "status": "Failure" if prediction_class == 1 else "No Failure",
            "issues_detected": {
                "Rotor & Blades": rotor_status,
                "Gearbox & Bearings": gearbox_status,
                "Generator": generator_status,
                "Temperature Sensors": temperature_status,
                "Event Logs": event_log_status
            },
            "overall_prediction": overall_status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Route to View All Stored Predictions
@app.route('/history', methods=['GET'])
def view_history():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions")
        rows = cursor.fetchall()

        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "rpm": row[1],
                "torque": row[2],
                "power_out": row[3],
                "t1": row[4], "t2": row[5], "t3": row[6],
                "event_count": row[7], "windspeed_ref": row[8],
                "voltage_L1": row[9], "voltage_L2": row[10],
                "current_out": row[11], "event_status": row[12],
                "turbine_status": row[13],
                "failure_detected": row[14],
                "rotor_status": row[15],
                "gearbox_status": row[16],
                "generator_status": row[17],
                "temperature_status": row[18],
                "event_log_status": row[19],
                "overall_status": row[20]
            })

    return jsonify({"history": history})

# ✅ Route to Delete All History
@app.route('/delete_history', methods=['DELETE'])
def delete_history():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
    return jsonify({"message": "All history deleted successfully"}), 200

# ✅ Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
