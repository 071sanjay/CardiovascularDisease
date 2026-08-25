# API
'''
Types of Requests
------------------
Get -> Read / Select
Post -> Create / Insert / Send
Put -> Update
Delete -> Remove
'''

from fastapi import FastAPI
from app.schema import CardioSchema
from app.model import load_logistic_model
import pandas as pd

# Fast API Object
app = FastAPI()
model, scaler = load_logistic_model()

# API Endpoints / Requests
@app.get('/')
def home():
    return 'Welcome to cardiovascular disease prediction'

@app.post('/predict-cardio-logistic')
def predict_cardio(data:CardioSchema):
    input_data = pd.DataFrame([
        # Accept data as JSON format
        data.model_dump()
    ])
    input_scaler = scaler.transform(input_data)
    prediction = model.predict(input_scaler)[0]
    return {
        'Prediction Status': int(prediction),
        'Status': 'Likely to be Healthy' if prediction == 0 else 'Likely to be Unhealthy'
    }
