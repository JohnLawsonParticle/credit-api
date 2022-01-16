# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 21:11:26 2022

@author: Guillaume
"""

import uvicorn
from fastapi import FastAPI
import joblib
import xgboost

import pandas as pd
 
app = FastAPI(debug=True)

df = pd.read_csv('data_api.csv')

model = open('pipe_model.joblib','rb')
clf = joblib.load(model)


@app.get('/')
async def index() : 
    return {"text" : "Hello World!"}


@app.get('/predictions')
async def get_predict(id_client) :
    resp = df[df['SK_ID_CURR']==int(id_client)]

    target = clf.predict(resp)[0]

    score = max(clf.predict_proba(resp)[0])
    
    if int(target) == 1 : 
        prediction = "Faulter"
    else : 
        prediction = "Non Faulter"
        
        
    return {"id" : id_client, "score" : str(score), "prediction" : prediction}

@app.get('/predictions')
async def get_predict_simu(id_client,new_credit_amt,new_credit_ann,new_income_percent,new_credit_term) :
    
    resp = df[df['SK_ID_CURR']==int(id_client)]
    
    resp["AMT_CREDIT"] = new_credit_amt
    resp["AMT_ANNUITY"] = new_credit_ann
    resp["CREDIT_INCOME_PERCENT"] = new_income_percent
    resp["CREDIT_TERM"] = new_credit_term
    
    target = clf.predict(resp)[0]

    score = max(clf.predict_proba(resp)[0])
    
    if int(target) == 1 : 
        prediction = "Faulter"
    else : 
        prediction = "Non Faulter"
        
        
    return {"id" : id_client, "score" : str(score), "prediction" : prediction}
    

if __name__ == '__main__' : 
    uvicorn.run(app, host="127.0.0.1",port=8000)
    
    
    
    