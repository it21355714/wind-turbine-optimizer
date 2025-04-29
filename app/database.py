import sqlite3
from datetime import datetime

def get_db():
    conn = sqlite3.connect("your_database.db")
    return conn  

def create_table():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            wind_speed REAL,
            wind_direction REAL,
            noise_level REAL,
            optimized_pitch_angle REAL,
            power_output REAL,
            rotor_speed REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(wind_speed, wind_direction, noise_level, optimized_pitch_angle, power_output, rotor_speed):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (
            wind_speed, wind_direction, noise_level, optimized_pitch_angle, 
            power_output, rotor_speed, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        wind_speed, wind_direction, noise_level,
        optimized_pitch_angle, power_output, rotor_speed,
        str(datetime.utcnow())
    ))
    conn.commit()
    conn.close()

def get_all_predictions(db):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM predictions")
    rows = cursor.fetchall()
    return rows


def delete_prediction(prediction_id):
    conn = get_db()
    cursor = conn.cursor()
    print(f"Executing DELETE query with id = {prediction_id}")
    cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
    conn.commit()
    deleted = cursor.rowcount
    print(f"Rows deleted: {deleted}")
    conn.close()
    return deleted > 0
