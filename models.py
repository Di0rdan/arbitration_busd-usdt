import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numpy.linalg as lg
# import eslr
from eslr import *

class IModel:
    def update(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError

    def get_stat(self, res):
        pass
    
    def features(self):
        return []
    
    def target(self):
        return []
    
class ESLR_diff_limit:
    
    def __init__(
        self, 
        comission, 
        mean_alpha, 
        features, 
        eslr_alpha_lm0,
        eslr_alpha_ml0,
        eslr_alpha_lm1,
        eslr_alpha_ml1,
        limit_lvl=1
    ):
        
        self.comission = comission
        self.alpha = mean_alpha
        self.eslr_features = features
        self.lvl = limit_lvl
        
        self.eslr0_lm = ExpSmoothLinearRegression(n=len(features), alpha=eslr_alpha_lm0)
        self.eslr0_ml = ExpSmoothLinearRegression(n=len(features), alpha=eslr_alpha_ml0)
        self.eslr1_lm = ExpSmoothLinearRegression(n=len(features), alpha=eslr_alpha_lm1)
        self.eslr1_ml = ExpSmoothLinearRegression(n=len(features), alpha=eslr_alpha_ml1)
        
        print(f'linear regression features: {self.eslr_features}')
        
    def update(self, X, y):
        self.eslr0_lm.update(
            np.array(X.loc[:, self.eslr_features], dtype=np.float64), 
            np.array(y[f'filled0_lm_lvl{self.lvl}'] - y.ask_pr_0, dtype=np.float64)
        )
        self.eslr0_ml.update(
            np.array(X.loc[:, self.eslr_features], dtype=np.float64), 
            np.array(y[f'filled0_ml_lvl{self.lvl}'] - y.bid_pr_1, dtype=np.float64)
        )
        self.eslr1_lm.update(
            np.array(X.loc[:, self.eslr_features], dtype=np.float64), 
            np.array(y[f'filled1_lm_lvl{self.lvl}'] - y.ask_pr_1, dtype=np.float64)
        )
        self.eslr1_ml.update(
            np.array(X.loc[:, self.eslr_features], dtype=np.float64), 
            np.array(y[f'filled1_ml_lvl{self.lvl}'] - y.bid_pr_0, dtype=np.float64)
        )
        
    def predict(self, X):
        
        diff0_lm = self.eslr0_lm.predict(np.array(X.loc[:, self.eslr_features], dtype=np.float64))
        diff0_ml = self.eslr0_ml.predict(np.array(X.loc[:, self.eslr_features], dtype=np.float64))
        
        diff1_lm = self.eslr1_lm.predict(np.array(X.loc[:, self.eslr_features], dtype=np.float64))
        diff1_ml = self.eslr1_ml.predict(np.array(X.loc[:, self.eslr_features], dtype=np.float64))
        
        lm0 = X.index[
            (X.ask_pr_1 + self.lvl) / (X.ask_pr_0 + diff0_lm) * \
            X[f'convert1_mean_{self.alpha}'] - 1.0 > self.comission
        ]
        
        lm1 = X.index[
            (X.ask_pr_0 + self.lvl) / (X.ask_pr_1 + diff1_lm) * \
            X[f'convert0_mean_{self.alpha}'] - 1.0 > self.comission
        ]
    
        ml0 = X.index[
            (X.bid_pr_1 + diff0_ml) / (X.bid_pr_0 - self.lvl) * \
            X[f'convert1_mean_{self.alpha}'] - 1.0 > self.comission
        ]
        ml1 = X.index[
            (X.bid_pr_0 + diff1_ml) / (X.bid_pr_1 - self.lvl) * \
            X[f'convert0_mean_{self.alpha}'] - 1.0 > self.comission
        ]
        
        return [], [], lm0, lm1, ml0, ml1
    
    def features(self):
        return list(
            set(self.eslr_features)
            .union(
                {
                    f'convert0_mean_{self.alpha}', 
                    f'convert1_mean_{self.alpha}',
                    'bid_pr_0',
                    'bid_pr_1',
                    'ask_pr_0',
                    'ask_pr_1',
                }
            )
        )
    
    def target(self):
        return [
            f'filled0_lm_lvl{self.lvl}',
            f'filled0_ml_lvl{self.lvl}',
            f'filled1_lm_lvl{self.lvl}',
            f'filled1_ml_lvl{self.lvl}',
            'bid_pr_0',
            'bid_pr_1',
            'ask_pr_0',
            'ask_pr_1',
        ]
    
    def get_stat(self, res):
        # print(f"volume market-market usdt->busd: {res['mm0'].shape[0]}")
        # print(f"volume market-market busd->usdt: {res['mm1'].shape[0]}")
        print(f"volume limit-market usdt->busd: {res['lm0'].shape[0]}")
        print(f"volume limit-market busd->usdt: {res['lm1'].shape[0]}")
        print(f"volume market-limit usdt->busd: {res['ml0'].shape[0]}")
        print(f"volume market-limit busd->usdt: {res['ml1'].shape[0]}")



        index=set()
        # index = index.union(set(res['mm0'].index))
        # index = index.union(set(res['mm1'].index))
        index = index.union(set(res['lm0'].index))
        index = index.union(set(res['ml0'].index))
        index = index.union(set(res['lm1'].index))
        index = index.union(set(res['ml1'].index))
        index = sorted(index)

        change0 = pd.Series(np.zeros(len(index)), index=index)
        change1 = pd.Series(np.zeros(len(index)), index=index)

        change1[res['lm0'].index] += \
            (res['lm0']['ask_pr_1'] + self.lvl) / (res['lm0'][f'filled0_lm_lvl{self.lvl}']) #* (1 - self.comission / 2)
        change0[res['lm0'].index] += -1.0
        change1[res['ml0'].index] += \
            res['ml0'][f'filled0_ml_lvl{self.lvl}'] / (res['ml0']['bid_pr_0'] - self.lvl) #* (1 - self.comission / 2)
        change0[res['ml0'].index] += -1.0

        change0[res['lm1'].index] += \
            (res['lm1']['ask_pr_0'] + self.lvl) / (res['lm1'][f'filled1_lm_lvl{self.lvl}']) #* (1 - self.comission / 2)
        change1[res['lm1'].index] += -1.0
        change0[res['ml1'].index] += \
            res['ml1'][f'filled1_ml_lvl{self.lvl}'] / (res['ml1']['bid_pr_1'] - self.lvl) #* (1 - self.comission / 2)
        change1[res['ml1'].index] += -1.0

        balance0 = change0.cumsum()
        balance1 = change1.cumsum()
        mask_plus = (balance0 > 0) & (balance1 > 0)


        convert0 = pd.Series(index=index)
        convert1 = pd.Series(index=index)

        convert0.loc[res['lm0'].index] = res['lm0'].bid_pr_1 / res['lm0'].ask_pr_0
        convert0.loc[res['ml0'].index] = res['ml0'].bid_pr_1 / res['ml0'].ask_pr_0
        convert0.loc[res['lm1'].index] = res['lm1'].bid_pr_1 / res['lm1'].ask_pr_0
        convert0.loc[res['ml1'].index] = res['ml1'].bid_pr_1 / res['ml1'].ask_pr_0

        convert1.loc[res['lm0'].index] = res['lm0'].bid_pr_0 / res['lm0'].ask_pr_1
        convert1.loc[res['ml0'].index] = res['ml0'].bid_pr_0 / res['ml0'].ask_pr_1
        convert1.loc[res['lm1'].index] = res['lm1'].bid_pr_0 / res['lm1'].ask_pr_1
        convert1.loc[res['ml1'].index] = res['ml1'].bid_pr_0 / res['ml1'].ask_pr_1


        converted0 =\
            balance0 + \
            balance1 * (balance1 < 0) / convert0 * (1 - self.comission) + \
            balance1 * (balance1 >= 0) * convert1 * (1 - self.comission)
        converted1 =\
            balance1 + \
            balance0 * (balance0 < 0) / convert1 * (1 - self.comission) + \
            balance0 * (balance0 >= 0) * convert0 * (1 - self.comission)

        time = (np.array(index) - index[0]) / 1000 / 60 / 60

        min_convert0 = converted0.rolling(int(time.shape[0] * 0.03 + 1)).min()
        min_convert1 = converted1.rolling(int(time.shape[0] * 0.03 + 1)).min()


        sharp0 = min_convert0[1:].diff().iloc[1:].mean() / np.sqrt(min_convert0.diff().iloc[1:].var())
        sharp1 = min_convert1[1:].diff().iloc[1:].mean() / np.sqrt(min_convert1.diff().iloc[1:].var())

        print(f'sharp in usdt: {sharp0}')
        print(f'sharp in busd: {sharp1}')

        fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
        fig.figsize=(16, 4)


        axs[0].plot(time, converted0, label='intime')
        axs[0].plot(time, min_convert0, c='red', label='last-3% min')
        axs[0].set_title('usdt')
        axs[0].set_xlabel('hours')

        axs[1].plot(time, converted1, label='intime')
        axs[1].plot(time, min_convert1, c='red', label='last-3% min')
        axs[1].set_title('busd')
        axs[1].set_xlabel('hours')

        fig.suptitle('intime profit', fontsize=16)
        # fig.savefig('pics/intime_profit.png')
        fig.show()

        fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
        fig.figsize=(16, 4)

        axs[0].plot(time, balance0)
        axs[0].scatter(time[mask_plus], balance0[mask_plus], 
                       s=40, linewidths=0.5, edgecolors='black', c='red', label='both plus')
        axs[0].set_title('usdt')
        axs[0].set_xlabel('hours')
        axs[0].legend()


        axs[1].plot(time, balance1)
        axs[1].scatter(time[mask_plus], balance1[mask_plus], 
                       s=40, linewidths=0.5, edgecolors='black', c='red', label='both plus')
        axs[1].set_title('busd')
        axs[1].set_xlabel('hours')
        axs[1].legend()


        fig.suptitle('balances', fontsize=16)
        # fig.savefig('pics/balances.png')
        fig.show()

        # ml 
        
        index=set()
        # index = index.union(set(res['mm0'].index))
        # index = index.union(set(res['mm1'].index))
#         index = index.union(set(res['lm0'].index))
        index = index.union(set(res['ml0'].index))
#         index = index.union(set(res['lm1'].index))
        index = index.union(set(res['ml1'].index))
        index = sorted(index)

        change0 = pd.Series(np.zeros(len(index)), index=index)
        change1 = pd.Series(np.zeros(len(index)), index=index)

#         change1[res['lm0'].index] += \
#             (res['lm0']['ask_pr_1'] + self.lvl) / (res['lm0'][f'filled0_lm_lvl{self.lvl}']) * \
#             (1 - self.comission / 2)
#         change0[res['lm0'].index] += -1.0
        change1[res['ml0'].index] += \
            res['ml0'][f'filled0_ml_lvl{self.lvl}'] / (res['ml0']['bid_pr_0'] - self.lvl) * \
            (1 - self.comission / 2)
        change0[res['ml0'].index] += -1.0

#         change0[res['lm1'].index] += \
#             (res['lm1']['ask_pr_0'] + self.lvl) / (res['lm1'][f'filled1_lm_lvl{self.lvl}']) * \
#             (1 - self.comission / 2)
#         change1[res['lm1'].index] += -1.0
        change0[res['ml1'].index] += \
            res['ml1'][f'filled1_ml_lvl{self.lvl}'] / (res['ml1']['bid_pr_1'] - self.lvl) * \
            (1 - self.comission / 2)
        change1[res['ml1'].index] += -1.0

        balance0 = change0.cumsum()
        balance1 = change1.cumsum()
        mask_plus = (balance0 > 0) & (balance1 > 0)


        convert0 = pd.Series(index=index)
        convert1 = pd.Series(index=index)

#         convert0.loc[res['lm0'].index] = res['lm0'].bid_pr_1 / res['lm0'].ask_pr_0
        convert0.loc[res['ml0'].index] = res['ml0'].bid_pr_1 / res['ml0'].ask_pr_0
#         convert0.loc[res['lm1'].index] = res['lm1'].bid_pr_1 / res['lm1'].ask_pr_0
        convert0.loc[res['ml1'].index] = res['ml1'].bid_pr_1 / res['ml1'].ask_pr_0

#         convert1.loc[res['lm0'].index] = res['lm0'].bid_pr_0 / res['lm0'].ask_pr_1
        convert1.loc[res['ml0'].index] = res['ml0'].bid_pr_0 / res['ml0'].ask_pr_1
#         convert1.loc[res['lm1'].index] = res['lm1'].bid_pr_0 / res['lm1'].ask_pr_1
        convert1.loc[res['ml1'].index] = res['ml1'].bid_pr_0 / res['ml1'].ask_pr_1


        converted0 =\
            balance0 + \
            balance1 * (balance1 < 0) / convert0 * (1 - self.comission) + \
            balance1 * (balance1 >= 0) * convert1 * (1 - self.comission)
        converted1 =\
            balance1 + \
            balance0 * (balance0 < 0) / convert1 * (1 - self.comission) + \
            balance0 * (balance0 >= 0) * convert0 * (1 - self.comission)

        time = (np.array(index) - index[0]) / 1000 / 60 / 60

        min_convert0 = converted0.rolling(int(time.shape[0] * 0.03 + 1)).min()
        min_convert1 = converted1.rolling(int(time.shape[0] * 0.03 + 1)).min()


        sharp0 = min_convert0[1:].diff().iloc[1:].mean() / np.sqrt(min_convert0.diff().iloc[1:].var())
        sharp1 = min_convert1[1:].diff().iloc[1:].mean() / np.sqrt(min_convert1.diff().iloc[1:].var())

        print(f'sharp in usdt: {sharp0}')
        print(f'sharp in busd: {sharp1}')

        fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
        fig.figsize=(16, 4)


        axs[0].plot(time, converted0, label='intime')
        axs[0].plot(time, min_convert0, c='red', label='last-3% min')
        axs[0].set_title('usdt')
        axs[0].set_xlabel('hours')

        axs[1].plot(time, converted1, label='intime')
        axs[1].plot(time, min_convert1, c='red', label='last-3% min')
        axs[1].set_title('busd')
        axs[1].set_xlabel('hours')

        fig.suptitle('intime profit', fontsize=16)
        # fig.savefig('pics/intime_profit.png')
        fig.show()

        fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
        fig.figsize=(16, 4)

        axs[0].plot(time, balance0)
        axs[0].scatter(time[mask_plus], balance0[mask_plus], 
                       s=40, linewidths=0.5, edgecolors='black', c='red', label='both plus')
        axs[0].set_title('usdt')
        axs[0].set_xlabel('hours')
        axs[0].legend()


        axs[1].plot(time, balance1)
        axs[1].scatter(time[mask_plus], balance1[mask_plus], 
                       s=40, linewidths=0.5, edgecolors='black', c='red', label='both plus')
        axs[1].set_title('busd')
        axs[1].set_xlabel('hours')
        axs[1].legend()


        fig.suptitle('balances', fontsize=16)
        # fig.savefig('pics/balances.png')
        fig.show()





    
class ESLR_MeanSolver:
    
    def __init__(self, comission, mean_alpha, features, eslr_alpha):
        self.comission = comission
        self.alpha = mean_alpha
        self.eslr_features = features
        self.eslr0 = ExpSmoothLinearRegression(n=len(features), alpha=eslr_alpha)
        self.eslr1 = ExpSmoothLinearRegression(n=len(features), alpha=eslr_alpha)
        print(f'linear regression features: {self.eslr_features}')
        
    def update(self, X, y):
        self.eslr0.update(
            np.array(X.loc[:, self.eslr_features], dtype=np.float64), 
            np.array(y.delay_m_convert0, dtype=np.float64)
        )
        self.eslr1.update(
            np.array(X.loc[:, self.eslr_features], dtype=np.float64), 
            np.array(y.delay_m_convert1, dtype=np.float64)
        )
        
    def predict(self, X):
        pred0 = self.eslr0.predict(np.array(X.loc[:, self.eslr_features], dtype=np.float64))
        pred1 = self.eslr1.predict(np.array(X.loc[:, self.eslr_features], dtype=np.float64))
        
        mm0 = X.index[pred0 * X[f'convert1_mean_{self.alpha}'] - 1.0 > self.comission]
        mm1 = X.index[pred1 * X[f'convert0_mean_{self.alpha}'] - 1.0 > self.comission]
        
        return mm0, mm1, [], [], [], []
    
        
    def features(self):
        return list(
            set(self.eslr_features)
            .union(
                {
                    f'convert0_mean_{self.alpha}', 
                    f'convert1_mean_{self.alpha}'
                }
            )
        )
    
    def target(self):
        return [
            'delay_m_convert0',
            'delay_m_convert1'
        ] 
    
    def get_stat(self, res):
        print(f"volume market-market usdt->busd: {res['mm0'].shape[0]}")
        print(f"volume market-market busd->usdt: {res['mm1'].shape[0]}")
        
        index=set()
        index = index.union(set(res['mm0'].index))
        index = index.union(set(res['mm1'].index))
        index = sorted(index)

        change0 = pd.Series(np.zeros(len(index)), index=index)
        change1 = pd.Series(np.zeros(len(index)), index=index)

        change1[res['mm0'].index] += res['mm0'].delay_m_convert0 #
        change0[res['mm0'].index] += -1.0

        change0[res['mm1'].index] += res['mm1'].delay_m_convert1 #
        change1[res['mm1'].index] += -1.0

        balance0 = change0.cumsum()
        balance1 = change1.cumsum()
        mask_plus = (balance0 > 0) & (balance1 > 0)


        convert0 = pd.Series(index=index)
        convert1 = pd.Series(index=index)

        convert0.loc[res['mm0'].index] = res['mm0'].convert0_mm
        convert0.loc[res['mm1'].index] = res['mm1'].convert1_mm
        convert1.loc[res['mm0'].index] = res['mm0'].convert0_mm
        convert1.loc[res['mm1'].index] = res['mm1'].convert1_mm


        converted0 =\
            balance0 + \
            balance1 * (balance1 < 0) / convert0 * (1 - self.comission) + \
            balance1 * (balance1 >= 0) * convert1 * (1 - self.comission)
        converted1 =\
            balance1 + \
            balance0 * (balance0 < 0) / convert1 * (1 - self.comission) + \
            balance0 * (balance0 >= 0) * convert0 * (1 - self.comission)

        time = (np.array(index) - index[0]) / 1000 / 60 / 60

        min_convert0 = converted0.rolling(int(time.shape[0] * 0.03)).min()
        min_convert1 = converted1.rolling(int(time.shape[0] * 0.03)).min()
                                       
        
        sharp0 = min_convert0[1:].diff().iloc[1:].mean() / np.sqrt(min_convert0.diff().iloc[1:].var())
        sharp1 = min_convert1[1:].diff().iloc[1:].mean() / np.sqrt(min_convert1.diff().iloc[1:].var())

        print(f'sharp in usdt: {sharp0}')
        print(f'sharp in busd: {sharp1}')
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
        fig.figsize=(16, 4)
        
        

        axs[0].plot(time, converted0, label='intime')
        axs[0].plot(time, min_convert0, c='red', label='last-3% min')
        axs[0].set_title('usdt')
        axs[0].set_xlabel('hours')
        
        axs[1].plot(time, converted1, label='intime')
        axs[1].plot(time, min_convert1, c='red', label='last-3% min')
        axs[1].set_title('busd')
        axs[1].set_xlabel('hours')
        
        fig.suptitle('intime profit', fontsize=16)
        # fig.savefig('pics/intime_profit.png')
        fig.show()

        fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
        fig.figsize=(16, 4)

        axs[0].plot(time, balance0)
        axs[0].scatter(time[mask_plus], balance0[mask_plus], 
                       s=40, linewidths=0.5, edgecolors='black', c='red', label='both plus')
        axs[0].set_title('usdt')
        axs[0].set_xlabel('hours')
        axs[0].legend()


        axs[1].plot(time, balance1)
        axs[1].scatter(time[mask_plus], balance1[mask_plus], 
                       s=40, linewidths=0.5, edgecolors='black', c='red', label='both plus')
        axs[1].set_title('busd')
        axs[1].set_xlabel('hours')
        axs[1].legend()


        fig.suptitle('balances', fontsize=16)
        # fig.savefig('pics/balances.png')
        fig.show()
        
class MeanSolver(IModel):
    
    def __init__(self, comission, alpha):
        self.comission = comission
        self.alpha = alpha
        
    def update(self, X, y):
        pass
    
    def predict(self, X):
        mm0 = X.index[X.convert0_mm * X[f'convert1_mean_{self.alpha}'] - 1.0 > self.comission]
        mm1 = X.index[X.convert1_mm * X[f'convert0_mean_{self.alpha}'] - 1.0 > self.comission]
        
        return mm0, mm1, [], [], [], []
    
    def features(self):
        return [
            'convert0_mm',
            'convert1_mm',
            f'convert0_mean_{self.alpha}',
            f'convert1_mean_{self.alpha}',
        ]
    
    def target(self):
        return [
            'delay_m_convert0',
            'delay_m_convert1'
        ]
    
    def get_stat(self, res):
        print(f"volume market-market usdt->busd: {res['mm0'].shape[0]}")
        print(f"volume market-market busd->usdt: {res['mm1'].shape[0]}")
        
        index=set()
        index = index.union(set(res['mm0'].index))
        index = index.union(set(res['mm1'].index))
        index = sorted(index)

        change0 = pd.Series(np.zeros(len(index)), index=index)
        change1 = pd.Series(np.zeros(len(index)), index=index)

        change1[res['mm0'].index] += res['mm0'].delay_m_convert0 #* (1 - self.comission)
        change0[res['mm0'].index] += -1.0

        change0[res['mm1'].index] += res['mm1'].delay_m_convert1 #* (1 - self.comission)
        change1[res['mm1'].index] += -1.0

        balance0 = change0.cumsum()
        balance1 = change1.cumsum()
        mask_plus = (balance0 > 0) & (balance1 > 0)


        convert0 = pd.Series(index=index)
        convert1 = pd.Series(index=index)

        convert0.loc[res['mm0'].index] = res['mm0'].convert0_mm
        convert0.loc[res['mm1'].index] = res['mm1'].convert1_mm
        convert1.loc[res['mm0'].index] = res['mm0'].convert0_mm
        convert1.loc[res['mm1'].index] = res['mm1'].convert1_mm


        converted0 =\
            balance0 + \
            balance1 * (balance1 < 0) / convert0 * (1 - self.comission) + \
            balance1 * (balance1 >= 0) * convert1 * (1 - self.comission)
        converted1 =\
            balance1 + \
            balance0 * (balance0 < 0) / convert1 * (1 - self.comission) + \
            balance0 * (balance0 >= 0) * convert0 * (1 - self.comission)

        time = (np.array(index) - index[0]) / 1000 / 60 / 60

        min_convert0 = converted0.rolling(int(time.shape[0] * 0.03)).min()
        min_convert1 = converted1.rolling(int(time.shape[0] * 0.03)).min()
                                       
        
        sharp0 = min_convert0[1:].diff().iloc[1:].mean() / np.sqrt(min_convert0.diff().iloc[1:].var())
        sharp1 = min_convert1[1:].diff().iloc[1:].mean() / np.sqrt(min_convert1.diff().iloc[1:].var())

        print(f'sharp in usdt: {sharp0}')
        print(f'sharp in busd: {sharp1}')
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
        fig.figsize=(16, 4)
        
        

        axs[0].plot(time, converted0, label='intime')
        axs[0].plot(time, min_convert0, c='red', label='last-3% min')
        axs[0].set_title('usdt')
        axs[0].set_xlabel('hours')
        
        axs[1].plot(time, converted1, label='intime')
        axs[1].plot(time, min_convert1, c='red', label='last-3% min')
        axs[1].set_title('busd')
        axs[1].set_xlabel('hours')
        
        fig.suptitle('intime profit', fontsize=16)
        # fig.savefig('pics/intime_profit.png')
        fig.show()

        fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=400)
        fig.figsize=(16, 4)

        axs[0].plot(time, balance0)
        axs[0].scatter(time[mask_plus], balance0[mask_plus], 
                       s=40, linewidths=0.5, edgecolors='black', c='red', label='both plus')
        axs[0].set_title('usdt')
        axs[0].set_xlabel('hours')
        axs[0].legend()


        axs[1].plot(time, balance1)
        axs[1].scatter(time[mask_plus], balance1[mask_plus], 
                       s=40, linewidths=0.5, edgecolors='black', c='red', label='both plus')
        axs[1].set_title('busd')
        axs[1].set_xlabel('hours')
        axs[1].legend()


        fig.suptitle('balances', fontsize=16)
        # fig.savefig('pics/balances.png')
        fig.show()
        
        