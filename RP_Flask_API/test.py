from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/SLIIT_lstm_wind_turbine_model.keras")

# Print model summary
model.summary()