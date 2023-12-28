import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from factor_manager.factor_template import FactorManager
from utils.cal_tools import ts_rolling_max, ts_rolling_min

class KDJ(FactorManager):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.source_path = '/root/HappyQuantTest/happyquant/raw_data/stock_index'
        self.target_path = '/root/HappyQuantTest/happyquant/factors/stock_index/kdj'

    def init_factors(
            self, 
            columns=['trade_date', 'close', 'high', 'low'],
            window_list=[9, 14, 21, 28],
        ):
        df = self.df_raw[columns].copy()

        self.df_factors = df['trade_date'].to_frame()
        for n in window_list:  
            self.df_factors['kdj_k_d_' + str(n)], self.df_factors['kdj_minus_j_' + str(n)],  self.df_factors['kdj_j_diff_' + str(n)] = cal_kdj(df, n)

    def update_factors(self):
        pass

def cal_rsv(df, n):
    return (df['close'] - ts_rolling_min(df['low'], n)) / (ts_rolling_max(df['high'], n) - ts_rolling_min(df['low'], n)) * 100

def cal_kdj(df, n, m1=3, m2=3):
    rsv = cal_rsv(df, n)
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k-d, -j, j.diff()