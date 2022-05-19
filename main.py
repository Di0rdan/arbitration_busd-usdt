import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numpy.linalg as lg
from make_signal import *

# model = ESLR_MeanSolver(
#     comission=0.00036,
#     mean_alpha=0.00001,
#     eslr_alpha=0.05,
#     features=[
#         'convert0_mm',
#         'convert1_mm',
#         'bid_vol_0',
#         'bid_vol_1',
#         'ask_vol_0',
#         'ask_vol_1'
#     ]
# )

# model = ESLR_diff_limit(
#     comission=0.0001, 
#     mean_alpha=0.00001, 
#     features=[
#         'convert0_mm',
#         'convert1_mm',
        
#         'bid_vol_0',
#         'bid_vol_1',
#         'ask_vol_0',
#         'ask_vol_1',
        
# #         'bid_pr_0',
# #         'bid_pr_1',
# #         'ask_pr_0',
# #         'ask_pr_1',
#     ], 
#     eslr_alpha_lm0=0.05,
#     eslr_alpha_ml0=0.05,
#     eslr_alpha_lm1=0.05,
#     eslr_alpha_ml1=0.05,
#     limit_lvl=1
# )

# model = MeanSolver(
#     comission=0.00036,
#     alpha=0.000001,
# )

data_manager = DataManager(
    # df=pd.read_csv('limit_targets.csv'),
    df=pd.read_csv('features.csv'),
    columns=list(set(model.features()).union(set(model.target()))),
    delay_m=20,
    delay_l=0,
    delay_order=1000
)

res = make_signals(
    data_manager,
    model,
    features=model.features(),
    look_ahead=model.target(),
    start=1649797200325, 
    finish=1649980799993,
#     finish=1649797200325+1000 * 60 * 60,
    history_time=1000 * 60 * 60 * 4,
    order_freq=50,
    step_size=1000 * 60 * 5,
)

model.get_stat(res)
