#Group 3 Final Project API Script - Jo-Anna Martinez, Amoya Jordan, Donate Tracey & Jadian Tulloch
from fastapi import FastAPI
from pydantic import BaseModel
import pickle 
import pandas as pd 

app = FastAPI()

#Load pipeline 
pipeline = pickle.load(open('rf_pipeline.pkl', 'rb'))

#Pydantic model for request body
class PredictRequest(BaseModel):
    port_of_entry: str
    type_of_visit: str
    origin: str
    month: str

@app.get ("/")
def root():
        return{"message": "Jamaica Tourism Forecasting API"}
    
@app.post("/predict")
def predict(request: PredictRequest):
        
        #create dataframe with same column names as training data(X)
        df_input = pd.DataFrame ([{ 
            "Port of Entry": request.port_of_entry,
            "Type of Visit": request.type_of_visit,
            "Origin": request.origin, 
            "Month": request.month
        }])
            

        prediction = pipeline.predict(df_input)[0]
        return {"predicted_number_of_visitors": float(prediction)}