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


# id_client = 104460


# resp = df[df['SK_ID_CURR']==int(id_client)]

# print(resp)


@app.get('/')
async def index() : 
    return {"text" : "Hello World!"}


@app.get('/predictions')
async def get_predict(id_client) :
    resp = df[df['SK_ID_CURR']==int(id_client)]

    target = clf.predict(resp)[0]

    score = clf.predict_proba(resp)[0][0]
    
    if int(target) == 1 : 
        prediction = "Faulter"
    else : 
        prediction = "Non Faulter"
        
        
    return {"id" : id_client, "score" : str(score), "prediction" : prediction}

if __name__ == '__main__' : 
    uvicorn.run(app, host="127.0.0.1",port=8000)
    
    
    
    