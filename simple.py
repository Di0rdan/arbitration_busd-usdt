import pandas as pd
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ExpSmoother:
    
    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.prev = None
    
    def update(self, x):
        if self.prev is None:
            res = pd.Series(x).ewm(alpha=self.alpha, adjust=False).mean()
        else:
            res = pd.concat((pd.Series([self.prev]), pd.Series(x))).ewm(alpha=self.alpha, adjust=False).mean()
        self.prev = res.iloc[res.shape[0] - 1]
        return res

class IModel:
    def update(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError


class Baseline(IModel):
    def __init__(self, **kwargs):
        
        self.real0 = []
        self.real1 = []
        
        self.pred0 = []
        self.pred1 = []
    
    
    def update(self, X, y):
        real0 = np.array(y.delay_m_profit0)
        real1 = np.array(y.delay_m_profit1)
        self.real0.append(real0)
        self.real1.append(real1)
        
    
    def predict(self, X):
        
        pred0 = X.profit0
        pred1 = X.profit1
        
        self.pred0.append(pred0)
        self.pred1.append(pred1)
        
        ans = pd.DataFrame(columns=['signal0', 'signal1'], dtype=np.int64)
        ans.signal0 = (X.profit0 > 0)
        ans.signal1 = (X.profit1 > 0)
        
        return ans
    
    def get_real(self):
        return np.concatenate(self.real0), np.concatenate(self.real1)
    
    def get_pred(self):
        return np.concatenate(self.pred0), np.concatenate(self.pred1)
    

class ExpSmoothLinearRegression:
    
    def __init__(self, n, alpha=0.5):
        self.alpha = alpha
        self.n = n
        self.XTX = np.eye(n) * 0.0
        self.XTy = np.zeros(n).reshape((n, 1))
    
    def update(self, x, y):
        self.XTX /= 1 - self.alpha
        self.XTy *= 1 - self.alpha
        x = x.reshape((self.n, 1)) * np.sqrt(self.alpha)
        
        self.XTX -= (self.XTX @ x) @ ((x.T @ self.XTX) / ((x.T @ self.XTX @ x)[0, 0] + 1.0))
        self.XTy += x * np.sqrt(self.alpha) * y
        
    def predict(self, x):
        
        return x.reshape((1, self.n)) @ self.XTX @ self.XTy

class ExpSmoothLinearRegression2:
    
    def __init__(self, n, alpha=0.5):
        self.alpha = alpha
        self.XTX = np.eye(n + 1)
        self.XTy = np.zeros((n + 1, 1))
        self.predictions = []
        
    def update(self, X, y):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y = y.reshape((X.shape[0], 1))
        
        self.XTX = self.XTX * (1 - self.alpha) + X.T @ X * self.alpha
        self.XTy = self.XTy * (1 - self.alpha) + X.T @ y * self.alpha
        
    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        pred = (X @ lg.inv(self.XTX) @ self.XTy).reshape(X.shape[0])
        self.predictions.append(pred)
        
        return pred
    
    def coef_(self):
        return (lg.inv(self.XTX) @ self.XTy).reshape((self.XTX.shape[0],))
    
class ESLR_Model(IModel):
    
    def __init__(self, n, alpha=0.5):
        self.lr0 = ExpSmoothLinearRegression2(n, alpha)
        self.lr1 = ExpSmoothLinearRegression2(n, alpha)
        
        self.real0 = []
        self.real1 = []
        
        self.pred0 = []
        self.pred1 = []
    
    def update(self, X, y):
        real0 = np.array(y.delay_m_profit0)
        real1 = np.array(y.delay_m_profit1)
        self.real0.append(real0)
        self.real1.append(real1)
        
        self.lr0.update(np.array(X), real0)
        self.lr1.update(np.array(X), real1)
        
    def predict(self, X):
        
        pred0 = self.lr0.predict(np.array(X))
        pred1 = self.lr1.predict(np.array(X))
        
        self.pred0.append(pred0)
        self.pred1.append(pred1)
        
        ans = pd.DataFrame(columns=['signal0', 'signal1'], dtype=np.int64)
        ans.signal0 = (pred0 > 0)
        ans.signal1 = (pred1 > 0)
        
        return ans
    
    def get_real(self):
        return np.concatenate(self.real0), np.concatenate(self.real1)
    
    def get_pred(self):
        return np.concatenate(self.pred0), np.concatenate(self.pred1)


import pandas as pd
import numpy as np

class DataManager:
    
    def __init__(self, df, delay_m, comission, features=[]):
        self.df = df
        self.df = self.df[~self.df.timestamp.duplicated(keep='last')]
        
        self.df['profit0'] = self.df.bid_pr_1 / self.df.ask_pr_0 * self.df.ratio_bid_pr * (1 - comission) - 1
        print(f'profit0 = bid1/ask0 * (1 - com) * r_bid - 1')
        self.df['profit1'] = self.df.bid_pr_0 / self.df.ask_pr_1 / self.df.ratio_ask_pr * (1 - comission) - 1
        print(f'profit1 = bid0/ask1 * (1 - com) / r_ask - 1')
            
        self.df['convert0'] = self.df.bid_pr_1 / self.df.ask_pr_0 * (1 - comission)
        print('convert0 = bid1/ask0 * (1 - com)')
        self.df['convert1'] = self.df.bid_pr_0 / self.df.ask_pr_1 * (1 - comission)
        print('convert1 = bid0/ask1 * (1 - com)')
        
        if 'adv_p0' in features:
            self.er0_smoother = ExpSmoother(alpha=0.00001)
            self.df['emp_ratio_0'] = (self.df.bid_pr_0 + self.df.ask_pr_0) / (self.df.bid_pr_1 + self.df.ask_pr_1)
        
        if 'adv_p1' in features:
            self.er1_smoother = ExpSmoother(alpha=0.00001)
            self.df['emp_ratio_1'] = (self.df.bid_pr_1 + self.df.ask_pr_1) / (self.df.bid_pr_0 + self.df.ask_pr_0)
        
        if 'profit0_normed' in features:
            self.p0_smoother = ExpSmoother(alpha=0.00001)
        if 'profit1_normed' in features:
            self.p1_smoother = ExpSmoother(alpha=0.00001)
        
        
        self.delay_m = delay_m
        self.comission = comission
        self.features = features
        
        print(features)
    
    def get(self, start, finish):
        cur_df = self.df[
            (self.df.timestamp >= start - 300 * 1000) & 
            (self.df.timestamp <= finish + 300 * 1000 + self.delay_m * 1000)
        ]
        cur_df.index = cur_df.timestamp // 1000
        
        cur_df = cur_df.reindex(np.arange(cur_df.index[0], cur_df.index[cur_df.shape[0] - 1] + 1)).ffill()
        cur_df.timestamp = cur_df.timestamp.astype(int)
            
        if 'adv_p0' in self.features:
            cur_df['er0_mean'] = self.er0_smoother.update(cur_df.emp_ratio_0)
            cur_df['adv_p0'] = cur_df.bid_pr_1 / cur_df.ask_pr_0 * cur_df.er0_mean - 1
        
        if 'adv_p1' in self.features:
            cur_df['er1_mean'] = self.er1_smoother.update(cur_df.emp_ratio_1)
            cur_df['adv_p1'] = cur_df.bid_pr_0 / cur_df.ask_pr_1 * cur_df.er1_mean - 1
            
        if 'profit0_normed' in self.features:
            cur_df['pr0_mean'] = self.p0_smoother.update(cur_df.profit0)
            cur_df['profit0_normed'] = cur_df.profit0 / cur_df.pr0_mean
        if 'profit1_normed' in self.features:
            cur_df['pr1_mean'] = self.p1_smoother.update(cur_df.profit1)
            cur_df['profit1_normed'] = cur_df.profit1 / cur_df.pr1_mean
        
        delay = cur_df[['profit0', 'profit1', 'convert0', 'convert1']].iloc[self.delay_m:, :]
        delay.index = cur_df.index[:-self.delay_m]
        
        cur_df = cur_df[
            list(
                filter(
                    lambda x: x not in ['ratio_bid_pr', 'ratio_ask_pr'],
                    self.features,
                )
            ) +
            ['ratio_bid_pr', 'ratio_ask_pr']
        ]

        cur_df['delay_m_profit0'] = delay.profit0
        cur_df['delay_m_profit1'] = delay.profit1
        cur_df['delay_m_convert0'] = delay.convert0
        cur_df['delay_m_convert1'] = delay.convert1
        
        cur_df = cur_df[(cur_df.index * 1000 >= start) & (cur_df.index * 1000 < finish)]
        cur_df.dropna(inplace=True)
        
        return cur_df



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
    start=1649797200325000, 
    finish=1649980799993000, 
    order_freq=50,
    step_size=1000 * 1000 * 60 * 5,
):
    signals0 = []
    signals1 = []
    res0 = []
    res1 = []
    
    step = 0
    for cur_start in range(start, finish, step_size):
        step += 1
#         print(f'step: {step}/{(finish - start) // step_size + 1}')
        cur_finish = min(finish, cur_start + step_size)
        data = data_manager.get(cur_start, cur_finish)
        X = data.loc[:, features]
        y = data[['delay_m_profit0', 'delay_m_profit1']]
        y_pred = model.predict(X=X)
        model.update(X=X, y=y)

        index0 = sparse_by_freq(data.index[y_pred.signal0], order_freq)
        res0.append(data.loc[index0, ['delay_m_convert0', 'ratio_bid_pr', 'ratio_ask_pr']])
        
        index1 = sparse_by_freq(data.index[y_pred.signal1], order_freq)
        res1.append(data.loc[index1, ['delay_m_convert1', 'ratio_bid_pr', 'ratio_ask_pr']])
        

    return pd.concat(res0, axis=0), pd.concat(res1, axis=0)
#     return res0, res1


df_raw = pd.read_csv('features.csv')

comission = 0.0003
delay = 40

print('init data manager...')
data_manager = DataManager(
    df_raw,
    delay_m=delay,
    comission=comission,
    features=features
)

print('init model...')
# model = ESLR_Model(n=len(features), alpha=0.05)
model = Baseline()

print('making signal...')
sig0, sig1 = make_signals(
    data_manager,
    model,
    features=features,
    order_freq=100
)

# print('graphing...')
# res = show_result(sig0, sig1)

inter = set(sig0.index).intersection(set(sig1.index))
sig0 = sig0.loc[sorted(set(sig0.index) - inter), :]
sig1 = sig1.loc[sorted(set(sig1.index) - inter), :]

res = pd.DataFrame(
    index=sorted(set(sig0.index).union(set(sig1.index))), 
    dtype=np.float64, 
    columns=[
        'change0', 'change1',
        'balance0', 'balance1',
        'converted0', 'converted1',
        'converted0_last', 'converted1_last',
        'r_bid', 'r_ask',
    ]
)

res.loc[sig0.index, 'r_bid'] = sig0.ratio_bid_pr
res.loc[sig1.index, 'r_bid'] = sig1.ratio_bid_pr

res.loc[sig0.index, 'r_ask'] = sig0.ratio_ask_pr
res.loc[sig1.index, 'r_ask'] = sig1.ratio_ask_pr

res = res[res.index > 1649797200325 + 1000 * 60 * 60 * 4]

res.loc[:, 'change0'] = sig0.delay_m_convert0
res.loc[:, 'change0'].fillna(-1.0, inplace=True)
res.loc[:, 'change1'] = sig1.delay_m_convert1
res.loc[:, 'change1'].fillna(-1.0, inplace=True)
res.loc[:, 'balance0'] = res.change0.cumsum()
res.loc[:, 'balance1'] = res.change1.cumsum()

res.converted0 =\
    res.balance0 + \
    res.balance1 * (res.balance1 < 0) * res.r_ask * (1 - comission / 2) + \
    res.balance1 * (res.balance1 >= 0) * res.r_bid * (1 - comission / 2)
res.converted1 =\
    res.balance1 + \
    res.balance0 * (res.balance0 < 0) / res.r_bid * (1 - comission / 2) + \
    res.balance0 * (res.balance0 >= 0) / res.r_ask * (1 - comission / 2)

res.converted0_last =\
    res.balance0 + \
    res.balance1 * (res.balance1 < 0) * res.r_ask.iloc[res.shape[0] - 1] * (1 - comission / 2) + \
    res.balance1 * (res.balance1 >= 0) * res.r_bid.iloc[res.shape[0] - 1] * (1 - comission / 2)
res.converted1_last =\
    res.balance1 + \
    res.balance0 * (res.balance0 < 0) / res.r_bid.iloc[res.shape[0] - 1] * (1 - comission / 2) + \
    res.balance0 * (res.balance0 >= 0) / res.r_ask.iloc[res.shape[0] - 1] * (1 - comission / 2)

fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
fig.figsize=(16, 4)

axs[0].set_title('converted0')
timestamp = (res.index - 1649797200325) / 1000 / 60 / 60
axs[0].plot(timestamp, res.converted0)
axs[0].set_xlabel('hours')

axs[1].set_title('converted1')
timestamp = (res.index - 1649797200325) / 1000 / 60 / 60
axs[1].plot(timestamp, res.converted1)
axs[1].set_xlabel('hours')

fig.savefig(
    '/Users/dmitrijvorobev/+iq/AMI/course_project_main/coursework/essay/'
    f'convert_{comission}_{delay}_.png'
)
fig.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
fig.figsize=(16, 4)

axs[0].set_title('converted0_last')
timestamp = (res.index - 1649797200325) / 1000 / 60 / 60
axs[0].plot(timestamp, res.converted0_last)
axs[0].set_xlabel('hours')

axs[1].set_title('converted1_last')
timestamp = (res.index - 1649797200325) / 1000 / 60 / 60
axs[1].plot(timestamp, res.converted1_last)
axs[1].set_xlabel('hours')

fig.savefig(
    '/Users/dmitrijvorobev/+iq/AMI/course_project_main/coursework/essay/'
    f'convert_last_{comission}_{delay}_.png'
)
fig.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
fig.figsize=(16, 4)

axs[0].set_title('balance0')
timestamp = (res.index - 1649797200325) / 1000 / 60 / 60
axs[0].plot(timestamp, res.balance0)
axs[0].set_xlabel('hours')

axs[1].set_title('balance1')
timestamp = (res.index - 1649797200325) / 1000 / 60 / 60
axs[1].plot(timestamp, res.balance1)
axs[1].set_xlabel('hours')

fig.savefig(
    '/Users/dmitrijvorobev/+iq/AMI/course_project_main/coursework/essay/'
    f'balance_{comission}_{delay}_.png'
)
fig.show()


real_0, real_1 = model.get_real()
pred_0, pred_1 = model.get_pred()

real_if_0 = real_0[real_0 > 0]
pred_if_0 = pred_0[real_0 > 0]
real_if_1 = real_1[real_0 > 0]
pred_if_1 = pred_1[real_0 > 0]

print(f'comission = {comission}')
print(f'delay = {delay}')
print()
print(f'sqrt(MSE_0)    = {np.sqrt(((real_0 - pred_0)**2).mean())}')
print(f'sqrt(MSE_IF_0) = {np.sqrt(((real_if_0 - pred_if_0)**2).mean())}')
print(f'MAE_0          = {np.abs(real_0 - pred_0).mean()}')
print(f'MAE_IF_0       = {np.abs(real_if_0 - pred_if_0).mean()}')
print()
print(f'sqrt(MSE_1)    = {np.sqrt(((real_1 - pred_1)**2).mean())}')
print(f'sqrt(MSE_IF_1) = {np.sqrt(((real_if_1 - pred_if_1)**2).mean())}')
print(f'MAE_1          = {np.abs(real_1 - pred_1).mean()}')
print(f'MAE_IF_1       = {np.abs(real_if_1 - pred_if_1).mean()}')

print()

print(f'sharp0: {res.converted0_last.diff().iloc[1:].mean() / np.sqrt(res.converted0_last.diff().iloc[1:].var())}')
print(f'sharp1: {res.converted1_last.diff().iloc[1:].mean() / np.sqrt(res.converted1_last.diff().iloc[1:].var())}')

