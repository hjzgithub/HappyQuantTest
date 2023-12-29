import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from factor_manager.factor_template import FactorManager
from utils.cal_tools import ts_rolling_mean

class Momentum(FactorManager):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.source_path = '/root/HappyQuantTest/happyquant/raw_data/stock_index'
        self.target_path = '/root/HappyQuantTest/happyquant/factors/stock_index/momentum'

    def init_factors(
            self, 
            columns=['trade_date', 'close', 'pre_close'],
            window_list=[1, 5, 10, 20, 40, 60],
        ):
        df = self.df_raw[columns].copy()
        df['rets'] = df['close'] / df['pre_close'] - 1
        self.df_factors = df['trade_date'].to_frame()
        for n in window_list:  
            self.df_factors['momentum_' + str(n)] = ts_rolling_mean(df['rets'], n)

    def update_factors(self):
        pass

