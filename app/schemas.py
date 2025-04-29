from pydantic import BaseModel
from typing import Optional

class PredictionInput(BaseModel):
    wind_speed: float
    wind_direction: float
    noise_level: Optional[float] = None

class PredictionOutput(BaseModel):
    rotor_speed: float
    power_output: float
    optimized_pitch_angle: float
    noise_level: Optional[float] = None

class PredictionSchema(PredictionInput):
    id: int
    timestamp: str
