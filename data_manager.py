import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numpy.linalg as lg

dependencies = {
    'convert0_mm' : {
        'bid_pr_1', 
        'ask_pr_0'
    },
    'convert1_mm' : {
        'bid_pr_0', 
        'ask_pr_1'
    },
    'delay_m_convert0' : {
        'bid_pr_1', 
        'ask_pr_0'
    },
    'delay_m_convert1' : {
        'bid_pr_0', 
        'ask_pr_1'
    },
}

for lvl in range(1, 6):
    dependencies[f'filled0_lm_lvl{lvl}'] = {
        'ask_pr_0',
        f'fillts_ask1_lvl{lvl}',
    }
    dependencies[f'filled0_ml_lvl{lvl}'] = {
        'bid_pr_1',
        f'fillts_bid0_lvl{lvl}',
    }
    dependencies[f'filled1_lm_lvl{lvl}'] = {
        'ask_pr_1',
        f'fillts_ask0_lvl{lvl}',
    }
    dependencies[f'filled1_ml_lvl{lvl}'] = {
        'bid_pr_0',
        f'fillts_bid1_lvl{lvl}',
    }

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

class DataManager:
    
    def __init__(
        self, 
        df, 
        columns=[],
        comission=None,
        delay_m=None,
        delay_l=None,
        delay_order=None
    ):
        

        
        print('reformat input data frame...')
        df.sort_index(inplace=True)
        df = df[~df.timestamp.duplicated(keep='last')]
        df.index = (df.timestamp // 1000).astype(int)
        
        print('init dependencies...')
        prep_columns = set()
        self.convert0_smoothers = []
        self.convert1_smoothers = []
        for i, feature in enumerate(columns):
            if feature in dependencies:
                prep_columns = prep_columns.union(dependencies[feature])
            elif feature in df.columns:
                prep_columns = prep_columns.union({feature})
            elif feature.startswith('convert0_mean'):
                prep_columns = prep_columns.union({'bid_pr_1', 'ask_pr_0'})
                alpha = float(feature[len('convert0_mean_'):])
                self.convert0_smoothers.append(ExpSmoother(alpha=alpha))
                columns[i] = f'convert0_mean_{alpha}'
            elif feature.startswith('convert1_mean'):
                pred_columns = prep_columns.union({'bid_pr_0', 'ask_pr_1'})
                alpha = float(feature[len('convert1_mean_'):])
                self.convert1_smoothers.append(ExpSmoother(alpha=alpha))
                columns[i] = f'convert1_mean_{alpha}'
            else:
                raise Execption(f'uncomleatable feature: {feature}')
        
        print('init storage...')
        self.df = df.loc[:, prep_columns]
        
        
        print('saving parameters...')
        self.columns = columns
        self.comission = comission
        self.delay_m = delay_m
        self.delay_l = delay_l
        self.delay_order = delay_order
        
    def get(self, start, finish):
        
        if self.delay_order is not None:
            cur_df = self.df[
                (self.df.index >= start - 300) & 
                (self.df.index <= finish + 300 + self.delay_m + self.delay_order)
            ]
        else:
            cur_df = self.df[
                (self.df.index >= start - 300) & 
                (self.df.index <= finish + 300 + self.delay_m)
            ]
            
        cur_df = cur_df.reindex(
            np.arange(cur_df.index[0], cur_df.index[cur_df.shape[0] - 1])
        ).ffill()
        cur_df = cur_df.reindex(
            np.arange(cur_df.index[0], max(finish + 5000, cur_df.index[cur_df.shape[0] - 1]))
        )
#         print(cur_df.index)
        
        data = pd.DataFrame(columns=self.columns, index=np.arange(start, finish))
        
        if 'bid_pr_1' in cur_df.columns and 'ask_pr_0' in cur_df.columns:
            convert0 = (cur_df.bid_pr_1 / cur_df.ask_pr_0)
        if 'bid_pr_0' in cur_df.columns and 'ask_pr_1' in cur_df.columns:
            convert1 = (cur_df.bid_pr_0 / cur_df.ask_pr_1)
       
        if 'bid_pr_0' in self.columns:
            data.loc[:, 'bid_pr_0'] = cur_df['bid_pr_0']
        if 'bid_pr_1' in self.columns:
            data.loc[:, 'bid_pr_1'] = cur_df['bid_pr_1']
        if 'ask_pr_0' in self.columns:
            data.loc[:, 'ask_pr_0'] = cur_df['ask_pr_0']
        if 'ask_pr_1' in self.columns:
            data.loc[:, 'ask_pr_1'] = cur_df['ask_pr_1']
        
        if 'bid_vol_0' in self.columns:
            data.loc[:, 'bid_vol_0'] = cur_df['bid_vol_0']
        if 'bid_vol_1' in self.columns:
            data.loc[:, 'bid_vol_1'] = cur_df['bid_vol_1']
        if 'ask_vol_0' in self.columns:
            data.loc[:, 'ask_vol_0'] = cur_df['ask_vol_0']
        if 'ask_vol_1' in self.columns:
            data.loc[:, 'ask_vol_1'] = cur_df['ask_vol_1']
        
        
        if 'convert0_mm' in self.columns:
            data.loc[:, 'convert0_mm'] = convert0
        if 'convert1_mm' in self.columns:
            data.loc[:, 'convert1_mm'] = convert1
        
        if 'delay_m_convert0' in self.columns:
            data.loc[:, 'delay_m_convert0'] = np.array(convert0[data.index + self.delay_m])
        if 'delay_m_convert1' in self.columns:
            data.loc[:, 'delay_m_convert1'] = np.array(convert1[data.index + self.delay_m])
            
        for feature in self.columns:
            if feature.startswith('filled0_lm'):
                lvl = int(feature[feature.find('lvl')+3:])
                convert_ts = (cur_df.loc[data.index + self.delay_l,  f'fillts_ask1_lvl{lvl}'] // 1000)
                convert_ts.index = data.index
                convert_ts = convert_ts[
                    (convert_ts != -1) &
                    ((convert_ts - convert_ts.index) < self.delay_order) &
                    (convert_ts + self.delay_m <= cur_df.index[cur_df.shape[0] - 1]) &
                    (convert_ts >= cur_df.index[0])
                    
                ] + self.delay_m
                
                data.loc[convert_ts.index, f'filled0_lm_lvl{lvl}'] = np.array(cur_df.loc[convert_ts, 'ask_pr_0'])
            if feature.startswith('filled0_ml'):
                lvl = int(feature[feature.find('lvl')+3:])
                convert_ts = (cur_df.loc[data.index + self.delay_l,  f'fillts_bid0_lvl{lvl}'] // 1000)
                convert_ts.index = data.index
                convert_ts = convert_ts[
                    (convert_ts != -1) &
                    ((convert_ts - convert_ts.index) < self.delay_order) &
                    (convert_ts + self.delay_m <= cur_df.index[cur_df.shape[0] - 1]) &
                    (convert_ts >= cur_df.index[0])
                ] + self.delay_m
                
                data.loc[convert_ts.index, f'filled0_ml_lvl{lvl}'] = np.array(cur_df.loc[convert_ts, 'bid_pr_1'])
            
            if feature.startswith('filled1_lm'):
                lvl = int(feature[feature.find('lvl')+3:])
                convert_ts = (cur_df.loc[data.index + self.delay_l,  f'fillts_ask0_lvl{lvl}'] // 1000)
                convert_ts.index = data.index
                convert_ts = convert_ts[
                    (convert_ts != -1) &
                    ((convert_ts - convert_ts.index) < self.delay_order) &
                    (convert_ts + self.delay_m <= cur_df.index[cur_df.shape[0] - 1]) &
                    (convert_ts >= cur_df.index[0])
                ] + self.delay_m
                
                data.loc[convert_ts.index, f'filled1_lm_lvl{lvl}'] = np.array(cur_df.loc[convert_ts, 'ask_pr_1'])
            if feature.startswith('filled1_ml'):
                lvl = int(feature[feature.find('lvl')+3:])
                convert_ts = (cur_df.loc[data.index + self.delay_l,  f'fillts_bid1_lvl{lvl}'] // 1000)
                convert_ts.index = data.index
                convert_ts = convert_ts[
                    (convert_ts != -1) &
                    ((convert_ts - convert_ts.index) < self.delay_order) &
                    (convert_ts + self.delay_m <= cur_df.index[cur_df.shape[0] - 1]) &
                    (convert_ts >= cur_df.index[0])
                ] + self.delay_m
                
                data.loc[convert_ts.index, f'filled1_ml_lvl{lvl}'] = np.array(cur_df.loc[convert_ts, 'bid_pr_0'])
                        
        for smoother in self.convert0_smoothers:
            data.loc[:, f'convert0_mean_{smoother.alpha}'] = smoother.update(convert0[data.index])
        for smoother in self.convert1_smoothers:
            data.loc[:, f'convert1_mean_{smoother.alpha}'] = smoother.update(convert1[data.index])
            
            
        data = data[(data.index >= start) & (data.index < finish)]
        
        return data

