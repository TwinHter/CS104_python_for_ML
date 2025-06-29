import datetime
import os
import traceback
import pandas as pd
import numpy as np
import pickle

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

# Biến toàn cục
model = None
avg_price_by_energy = None
avg_price_by_tenure = None
avg_price_by_property_type = None
avg_price_by_street = None
avg_price_by_incode = None
avg_price_by_outcode = None
global_mean_price = None

class InputItem(BaseModel):
    fullAddress: str
    outcode: str
    longitude: float
    latitude: float
    bedrooms: int
    floorAreaSqM: float
    livingRooms: int
    bathrooms: int
    tenure: str
    propertyType: str
    currentEnergyRating: str
    sale_month: int
    sale_year: int

def load_n_preprocess_data():
    global avg_price_by_energy, avg_price_by_tenure, avg_price_by_property_type
    global avg_price_by_street, avg_price_by_incode, avg_price_by_outcode
    global global_mean_price

    data_path = "/Users/henry/Documents/SE_training/app/Proj3/backend/data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data file not found: {data_path}")

    df_train = pd.read_csv(data_path)
    df_train.dropna(subset=['price'], inplace=True)

    avg_price_by_energy = df_train.groupby('currentEnergyRating')['price'].mean().to_dict()
    avg_price_by_tenure = df_train.groupby('tenure')['price'].mean().to_dict()
    avg_price_by_property_type = df_train.groupby('propertyType')['price'].mean().to_dict()
    avg_price_by_street = df_train.groupby('street')['price'].mean().to_dict()
    avg_price_by_incode = df_train.groupby('incode')['price'].mean().to_dict()
    avg_price_by_outcode = df_train.groupby('outcode')['price'].mean().to_dict()
    global_mean_price = df_train['price'].mean()

def get_models(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Model file not found: {filename}")
    with open(filename, 'rb') as f:
        return pickle.load(f)

def preprocess_input(input_item: InputItem) -> pd.DataFrame:
    data = input_item.dict()
    input_df = pd.DataFrame([data])

    input_df['street'] = input_df['fullAddress'].apply(
        lambda x: x.split(',')[-3].strip() if isinstance(x, str) and ',' in x else "unknown"
    )

    input_df['currentEnergyRating'] = input_df['currentEnergyRating'].map(avg_price_by_energy)
    input_df['currentEnergyRating'].fillna(global_mean_price, inplace=True)

    input_df['tenure'] = input_df['tenure'].map(avg_price_by_tenure)
    input_df['tenure'].fillna(global_mean_price, inplace=True)

    input_df['propertyType'] = input_df['propertyType'].map(avg_price_by_property_type)
    input_df['propertyType'].fillna(global_mean_price, inplace=True)

    input_df['street'] = input_df['street'].map(avg_price_by_street)
    input_df['street'].fillna(global_mean_price, inplace=True)

    input_df['outcode'] = input_df['outcode'].map(avg_price_by_outcode)
    input_df['outcode'].fillna(global_mean_price, inplace=True)

    input_df['rooms'] = (
        input_df['bedrooms'] * 2 +
        input_df['bathrooms'] * 1.6 +
        input_df['livingRooms'] * 3.01
    )

    input_df['sale_year'] = input_df['sale_year'] + input_df['sale_month'] / 12

    return input_df[[
        'outcode', 'latitude', 'longitude', 'bathrooms', 'bedrooms',
        'floorAreaSqM', 'livingRooms', 'tenure', 'propertyType',
        'currentEnergyRating', 'sale_year', 'street', 'rooms'
    ]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    load_n_preprocess_data()
    model = get_models("xgb_pipeline_model.pkl")
    print("✅ Models and data initialized (via lifespan)")
    yield
    # Cleanup logic (nếu cần)

app = FastAPI(lifespan=lifespan)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/model")
async def predict(input_item: InputItem):
    try:
        input_array = preprocess_input(input_item)
        predicted_price = float(model.predict(input_array)[0])  # convert numpy.float32 -> float

        chart_data = []
        current_month = input_item.sale_month
        current_year = input_item.sale_year

        input_dict = input_item.dict()
        for i in range(12):
            month = (current_month + i - 1) % 12 + 1
            year = current_year + (current_month + i - 1) // 12
            input_dict.update({"sale_month": month, "sale_year": year})
            updated_input = InputItem(**input_dict)
            updated_array = preprocess_input(updated_input)
            month_prediction = float(model.predict(updated_array)[0])  # convert here as well

            chart_data.append({
                "month": datetime.datetime(year, month, 1).strftime("%b %Y"),
                "price": round(month_prediction, 2)
            })

        prices = [m["price"] for m in chart_data]
        threshold_ratio = 0.03
        is_stable = bool(np.std(prices) < predicted_price * threshold_ratio)  # convert to Python bool

        return {
            "price": round(predicted_price, 2),
            "isStable": is_stable,
            "priceHistoryChartData": chart_data
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Prediction error: {str(e)}"}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
