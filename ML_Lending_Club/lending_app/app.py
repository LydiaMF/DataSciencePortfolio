import pandas as pd
from pydantic import BaseModel, Field
import pickle
import uvicorn
from fastapi import FastAPI


# Pydantic classes for input and output
class LoanApplication(BaseModel):
    Amount_Requested: int = Field(alias='Amount Requested')
    Risk_Score: int #= Field(alias='my key')
    Employment_Length: str = Field(alias='Employment Length')
    dti: float

    class Config:
        populate_by_name = True

class PredictionOut(BaseModel):
    loan_acceptance: str #float


# Load the model
pickle_in = open("loan_acceptance_Bagged_model.pickle","rb")
model=pickle.load(pickle_in)

#model = cb.CatBoostClassifier()
#model.load_model("loan_catboost_model.cbm")



# Start the app
app = FastAPI()


# Home page
@app.get("/")
def home():
    return {"message": "Loan Application Status Prediction App", "model_version": 0.1}


# Inference endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(data: LoanApplication):
    cust_df = pd.DataFrame([data.model_dump(by_alias=True)])
    pred = model.predict(cust_df)
    classes = {0: 'accepted', 1: 'rejected'}
    result = {"loan_acceptance": classes[pred[0]]}
    return result


# . Run the API with uvicorn
#    Will run on http://127.0.0.1:8000   or   http://0.0.0.0:8080 
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
    