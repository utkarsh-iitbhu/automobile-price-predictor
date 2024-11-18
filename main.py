from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from automobile import AutoMobile
import pandas  as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


app = FastAPI(title="Automobile Price Prediction API",
             description="API for predicting automobile prices using ensemble model",
             version="1.0.0")

class CarFeatures(BaseModel):
    fuel_type: str
    aspiration: str
    num_of_doors: str
    body_style: str
    drive_wheels: str
    engine_location: str
    wheel_base: float
    length: float
    width: float
    height: float
    curb_weight: float
    engine_type: str
    num_of_cylinders: str
    engine_size: int
    fuel_system: str
    bore: float
    stroke: float
    compression_ratio: float
    horsepower: int
    peak_rpm: int
    city_mpg: int
    highway_mpg: int

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # Render the index.html template
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_price(features: CarFeatures):
    try:
        model = AutoMobile()
        
        # Convert pydantic model to dict and rename keys to match training data
        input_data = {
            'fuel-type': features.fuel_type,
            'aspiration': features.aspiration,
            'num-of-doors': features.num_of_doors,
            'body-style': features.body_style,
            'drive-wheels': features.drive_wheels,
            'engine-location': features.engine_location,
            'wheel-base': features.wheel_base,
            'length': features.length,
            'width': features.width,
            'height': features.height,
            'curb-weight': features.curb_weight,
            'engine-type': features.engine_type,
            'num-of-cylinders': features.num_of_cylinders,
            'engine-size': features.engine_size,
            'fuel-system': features.fuel_system,
            'bore': features.bore,
            'stroke': features.stroke,
            'compression-ratio': features.compression_ratio,
            'horsepower': features.horsepower,
            'peak-rpm': features.peak_rpm,
            'city-mpg': features.city_mpg,
            'highway-mpg': features.highway_mpg
        }
        
        prediction = model.predict(input_data)
        return {"predicted_price": round(prediction, 2)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)