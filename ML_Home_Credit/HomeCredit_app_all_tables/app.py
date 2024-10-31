import pandas as pd
from pydantic import BaseModel, Field
import pickle
import uvicorn
from fastapi import FastAPI

from typing import Union

# Pydantic classes for input and output
class LoanApplication(BaseModel):
    
    #dti: Union[float, None] = Field(default=None, alias='dti')
    AMT_ANNUITY_CREDIT_ratio: Union[float, None] = Field(default = None, alias = 'AMT_ANNUITY_CREDIT_ratio')
    AMT_ANNUITY_CREDIT_ratio_prev: Union[float, None] = Field(default = None, alias = 'AMT_ANNUITY_CREDIT_ratio_prev')
    AMT_CREDIT_DEBT_LIMIT_RATIO: Union[float, None] = Field(default = None, alias = 'AMT_CREDIT_DEBT_LIMIT_RATIO')
    AMT_GOODS_PRICE: Union[float, None] = Field(default = None, alias = 'AMT_GOODS_PRICE')
    AMT_GOODS_PRICE_CREDIT_ratio_prev: Union[float, None] = Field(default = None, alias = 'AMT_GOODS_PRICE_CREDIT_ratio_prev')
    DAYS_FIRST_DUE: Union[float, None] = Field(default = None, alias = 'DAYS_FIRST_DUE')
    DAYS_LAST_DUE_1ST_VERSION: Union[float, None] = Field(default = None, alias = 'DAYS_LAST_DUE_1ST_VERSION')
    EXT_SOURCE_1: Union[float, None] = Field(default = None, alias = 'EXT_SOURCE_1')
    EXT_SOURCE_2: Union[float, None] = Field(default = None, alias = 'EXT_SOURCE_2')
    EXT_SOURCE_3: Union[float, None] = Field(default = None, alias = 'EXT_SOURCE_3')
    HOUR_APPR_PROCESS_START_prev: Union[float, None] = Field(default = None, alias = 'HOUR_APPR_PROCESS_START_prev')
    SELLERPLACE_AREA: Union[float, None] = Field(default = None, alias = 'SELLERPLACE_AREA')
    TERM_ACTUAL_months: Union[float, None] = Field(default = None, alias = 'TERM_ACTUAL_months')
    int_rate: Union[float, None] = Field(default = None, alias = 'int_rate')
    int_rate_plain: Union[float, None] = Field(default = None, alias = 'int_rate_plain')
    int_rate_plain_dif: Union[float, None] = Field(default = None, alias = 'int_rate_plain_diff')




    class Config:
        populate_by_name = True

class PredictionOut(BaseModel):
    payment_issue_probability: float
    best_threshold: float    
    result_for_threshold: str #float



# Load the model

filename = "defaultrisk_model_all_tables.pickle"
pickle_in = open(filename,"rb")
model = pickle.load(pickle_in)
model_thresh = 0.53



# Start the app
app = FastAPI()


# Home page
@app.get("/")
def home():
    return {"message": "Payment Issue Prediction App for Clients with extensive Credit History by LMF", "model_version": 0.1}


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
    