import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numpy.linalg as lg
# import models
from models import *

def sparse_by_freq(seq, order_freq):
    res = []
    last = -np.inf
    for value in seq:
        if value - order_freq >= last:
            last = value
            res.append(value)
    return res

def make_signals(
    data_manager,
    model,
    features,
    look_ahead,
    start=1649797200325, 
    finish=1649980799993, 
    history_time=1000 * 60 * 60 * 24,
    order_freq=50,
    step_size=1000 * 60 * 5,
):
    step = 0
    
    res_mm0 = []
    res_mm1 = []
    res_lm0 = []
    res_ml0 = []
    res_lm1 = []
    res_ml1 = []
    
    for cur_start in range(start, finish, step_size):
        step += 1
        print(f'step: {step}/{(finish - start) // step_size + 1}')
        
        cur_finish = min(finish, cur_start + step_size)
        
        data = data_manager.get(cur_start, cur_finish)
        data.dropna(inplace=True)
        X = data.loc[:, features]
        y = data.loc[:, look_ahead]
        
        if cur_start > start + history_time:
            mm0, mm1, lm0, ml0, lm1, ml1 = model.predict(X=X)
            res_mm0.append(data.loc[sparse_by_freq(mm0, order_freq), :])
            res_mm1.append(data.loc[sparse_by_freq(mm1, order_freq), :])
            res_lm0.append(data.loc[sparse_by_freq(lm0, order_freq), :])
            res_ml0.append(data.loc[sparse_by_freq(ml0, order_freq), :])
            res_lm1.append(data.loc[sparse_by_freq(lm1, order_freq), :])
            res_ml1.append(data.loc[sparse_by_freq(ml1, order_freq), :])

        model.update(X=X, y=y)
        
    return {
        'mm0' : pd.concat(res_mm0, axis=0),
        'mm1' : pd.concat(res_mm1, axis=0),
        'lm0' : pd.concat(res_lm0, axis=0),
        'ml0' : pd.concat(res_ml0, axis=0),
        'lm1' : pd.concat(res_lm1, axis=0),
        'ml1' : pd.concat(res_ml1, axis=0),
    }
