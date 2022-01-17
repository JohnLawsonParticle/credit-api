# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 21:11:26 2022

@author: Guillaume
"""

import uvicorn
from fastapi import FastAPI
import joblib
import xgboost
from sklearn.neighbors import NearestNeighbors

import pandas as pd
 
app = FastAPI(debug=True)

df = pd.read_csv('data_api.csv')

model = open('pipe_model.joblib','rb')
clf = joblib.load(model)

processor = open('processor.joblib','rb')
prep_pipe = joblib.load(processor)

df_trans = prep_pipe.fit_transform(df)

nbrs = NearestNeighbors(n_neighbors=5)
nbrs.fit(df_trans)

distances, indices = nbrs.kneighbors(df_trans)

def get_index_from_SK_ID(sk_id = df.loc[0,"SK_ID_CURR"]) :
    loc_id = df.loc[df["SK_ID_CURR"] == sk_id,"SK_ID_CURR"]
    index_user = loc_id.index.values[0]
    return index_user


@app.get('/')
async def index() : 
    return {"text" : "Hello World!"}


def predict(resp) : 
    
    target = clf.predict(resp)[0]

    score = clf.predict_proba(resp)[0][0]
    
    if int(target) == 1 : 
        prediction = "Faulter"
    else : 
        prediction = "Non Faulter"
        
        
    return {"score" : str(score), "prediction" : prediction}
    

@app.get('/get_predict')
async def get_predict(id_client) :
    
    resp = df[df['SK_ID_CURR']==int(id_client)]

    return predict(resp)

@app.get('/get_predict_simu')
async def get_predict_simu(id_client,new_credit_amt,new_credit_ann,new_income_percent,new_credit_term) :
    
    resp = df[df['SK_ID_CURR']==int(id_client)]
    
    resp["AMT_CREDIT"] = new_credit_amt
    resp["AMT_ANNUITY"] = new_credit_ann
    resp["CREDIT_INCOME_PERCENT"] = new_income_percent
    resp["CREDIT_TERM"] = new_credit_term
    
    return predict(resp)

@app.get('/get_neighbours')
async def get_neighbours(id_client) :
        
    neigh = indices[get_index_from_SK_ID(int(id_client))]
    neigh_1 = neigh[1]
    neigh_2 = neigh[2]
    neigh_3 = neigh[3]
    neigh_4 = neigh[4]
    
    return {"neigh_1" : int(neigh_1), "neigh_2" : int(neigh_2), "neigh_3" : int(neigh_3), "neigh_4" : int(neigh_4)}
  

if __name__ == '__main__' : 
    uvicorn.run(app, host="127.0.0.1",port=8000)
    
    
    
    