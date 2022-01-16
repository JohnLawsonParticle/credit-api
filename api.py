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

model = open('final_model_only.pkl','rb')
clf = joblib.load(model)

# clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.7,
#               enable_categorical=False, gamma=0, gpu_id=-1,
#               importance_type=None, interaction_constraints='',
#               learning_rate=0.3, max_delta_step=0, max_depth=2,
#               min_child_weight=3, monotone_constraints='()',
#               n_estimators=230, n_jobs=-1, num_parallel_tree=1,
#               predictor='auto', random_state=3, reg_alpha=0.001, reg_lambda=5,
#               scale_pos_weight=2.0, subsample=1, tree_method='auto',
#               validate_parameters=1, verbosity=0)

# print(clf)

clf.predict(df)

# id_client = 177250

# resp = df[df['SK_ID_CURR']==int(id_client)].reset_index()

# print(clf.predict(resp)[0])


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
    
    
    
    