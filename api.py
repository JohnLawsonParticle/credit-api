# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 21:11:26 2022

@author: Guillaume
"""

import uvicorn
from fastapi import FastAPI
import joblib


import pandas as pd
 
app = FastAPI(debug=True)

df = pd.read_csv('sampled_test_set_no_pred.csv')

model = open('final_model.pkl','rb')
clf = joblib.load(model)

@app.get('/')
async def index() : 
    return {"text" : "Hello World!"}


@app.get('/predictions')
async def get_predict(id_client) :
    resp = df[df['SK_ID_CURR']==int(id_client)].reset_index()

    target = clf.predict(resp)[0]

    score = clf.predict_proba(resp)[0][0]
    
    if int(target) == 1 : 
        prediction = "Faulter"
    else : 
        prediction = "Non Faulter"
        
        
    return {"id" : id_client, "score" : str(score), "prediction" : prediction}

if __name__ == '__main__' : 
    uvicorn.run(app, host="127.0.0.1",port=8000)
    
    
    
    