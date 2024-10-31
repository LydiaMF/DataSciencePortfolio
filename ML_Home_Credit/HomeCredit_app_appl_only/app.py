import pandas as pd
from pydantic import BaseModel, Field
import pickle
import uvicorn
from fastapi import FastAPI

from typing import Union

# Pydantic classes for input and output
class LoanApplication(BaseModel):
    
    AMT_ANNUITY_CREDIT_ratio: Union[float, None] = Field(default=None, alias='AMT_ANNUITY_CREDIT_ratio')
    AMT_INCOME_TOTAL: Union[float, None] = Field(default=None, alias='AMT_INCOME_TOTAL')
    APARTMENTS_MODE: Union[float, None] = Field(default=None, alias='APARTMENTS_MODE')
    BASEMENTAREA_MODE: Union[float, None] = Field(default=None, alias='BASEMENTAREA_MODE')
    COMMONAREA_MEDI: Union[float, None] = Field(default=None, alias='COMMONAREA_MEDI')
    DAYS_BIRTH: Union[int, None] = Field(default=None, alias='DAYS_BIRTH')
    DAYS_EMPLOYED: Union[int, None] = Field(default=None, alias='DAYS_EMPLOYED')
    DAYS_ID_PUBLISH: Union[int, None] = Field(default=None, alias='DAYS_ID_PUBLISH')
    DAYS_LAST_PHONE_CHANGE: Union[int, None] = Field(default=None, alias='DAYS_LAST_PHONE_CHANGE')  # shown as "float
    DAYS_REGISTRATION: Union[int, None] = Field(default=None, alias='DAYS_REGISTRATION')            # shown as "float
    ENTRANCES_AVG: Union[float, None] = Field(default=None, alias='ENTRANCES_AVG')
    EXT_RATIO: Union[float, None] = Field(default=None, alias='EXT_RATIO')
    EXT_SOURCE_1: Union[float, None] = Field(default=None, alias='EXT_SOURCE_1')
    EXT_SOURCE_2: Union[float, None] = Field(default=None, alias='EXT_SOURCE_2')
    EXT_SOURCE_3: Union[float, None] = Field(default=None, alias='EXT_SOURCE_3')
    EXT_SUM2: Union[float, None] = Field(default=None, alias='EXT_SUM2')
    FLOORSMIN_AVG: Union[float, None] = Field(default=None, alias='FLOORSMIN_AVG')
    HOUR_APPR_PROCESS_START: Union[int, None] = Field(default=None, alias='HOUR_APPR_PROCESS_START')
    LANDAREA_MODE: Union[float, None] = Field(default=None, alias='LANDAREA_MODE')
    LIVINGAPARTMENTS_MODE: Union[float, None] = Field(default=None, alias='LIVINGAPARTMENTS_MODE')
    LIVINGAREA_MODE: Union[float, None] = Field(default=None, alias='LIVINGAREA_MODE')
    NONLIVINGAPARTMENTS_AVG: Union[float, None] = Field(default=None, alias='NONLIVINGAPARTMENTS_AVG')
    NONLIVINGAREA_MODE: Union[float, None] = Field(default=None, alias='NONLIVINGAREA_MODE')
    OWN_CAR_AGE: Union[float, None] = Field(default=None, alias='OWN_CAR_AGE')
    REGION_POPULATION_RELATIVE: Union[float, None] = Field(default=None, alias='REGION_POPULATION_RELATIVE')
    YEARS_BEGINEXPLUATATION_AVG: Union[float, None] = Field(default=None, alias='YEARS_BEGINEXPLUATATION_AVG')
    YEARS_BUILD_MODE: Union[float, None] = Field(default=None, alias='YEARS_BUILD_MODE')
    building_avg: Union[float, None] = Field(default=None, alias='building_avg')
    dti: Union[float, None] = Field(default=None, alias='dti')


    class Config:
        populate_by_name = True

class PredictionOut(BaseModel):
    payment_issue_probability: float
    best_threshold: float    
    result_for_threshold: str #float



# Load the model

filename = "defaultrisk_model_app_only.pickle"
pickle_in = open(filename,"rb")
model = pickle.load(pickle_in)
model_thresh = 0.47



# Start the app
app = FastAPI()


# Home page
@app.get("/")
def home():
    return {"message": "Payment Issue Prediction App for Clients with no Credit History by LMF", "model_version": 0.1}


# Inference endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(data: LoanApplication):
    cust_df = pd.DataFrame([data.model_dump(by_alias=True)])
    prob = model.predict_proba(cust_df).round(3)
    threshold = model_thresh
    pred = (prob[:, 1] > threshold).astype("float")
    #pred = model.predict(cust_df)

    classes = {0: 'No Payment Issues', 1: 'Payment Issues'}
    result = {"payment_issue_probability": prob[:, 1], 
              "best_threshold": threshold, 
              "result_for_threshold": classes[pred[0]]}
    return result


# . Run the API with uvicorn
#    Will run on http://127.0.0.1:8000   or   http://0.0.0.0:8000 
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
    