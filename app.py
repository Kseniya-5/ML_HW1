from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

# Загрузка модели ElasticNet
model = joblib.load('C:\\Users\\ksbal\\PycharmProjects\\project1\\pythonProject\\model.joblib')

# Создаем объект FastApi
app = FastAPI()

class Processing:
    def __init__(self, data):
        if isinstance(data, dict):
            self.df = pd.DataFrame(data, index=[0])
        else:
            self.df = pd.DataFrame(data)

    def name(self):
        return self.df.rename(columns={'mileage (kmpl)': 'mileage', 'engine (CC)': 'engine', 'max_power (bhp)': 'max_power'},
                  inplace=True)
    def encod(self):
        self.df_encode = pd.get_dummies(self.df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True).astype(int)
        return self.df_encode

class Item(BaseModel):
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    seats: int


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # Подготовка данных
    input_data = item.dict()
    processing = Processing(input_data)
    df_processing = processing.name().encod()

    predicted_price = model.predict(df_processing)
    return float(predicted_price[0])


@app.post("/predict_items")
async def predict_items(items: List[Item]) -> List[float]:
    df = pd.read_csv('C:\\Users\\ksbal\\PycharmProjects\\project1\\pythonProject\\df.csv')

    # Обработка данных
    processing = Processing(df)
    df_processing = processing.name().encod()

    predicted_prices = model.predict(df_processing)
    df['predicted_price'] = predicted_prices

    # Сохраняем результат в новый CSV файл
    output_file_path = 'C:\\Users\\ksbal\\PycharmProjects\\project1\\pythonProject\\predictions.csv'
    df.to_csv(output_file_path, index=False)

    return {"message": "Predictions saved to predictions.csv", "predicted_prices": predicted_prices.tolist()}