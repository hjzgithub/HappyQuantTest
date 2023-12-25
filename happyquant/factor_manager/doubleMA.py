import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from factor_manager.factor_template import FactorManager
from utils.cal_tools import ts_rolling_mean

class DoubleMA(FactorManager):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.source_path = '/root/HappyQuantTest/happyquant/raw_data/stock_index'
        self.target_path = '/root/HappyQuantTest/happyquant/factors/stock_index/doubleMA'

    def init_factors(self):
        df = self.df_raw[['trade_date', 'close']].copy()
        self.df_factors = df['trade_date'].to_frame()
        self.df_factors['doubleMA_5_10'] = ts_rolling_mean(df['close'], 5) - ts_rolling_mean(df['close'], 10)
        self.df_factors['doubleMA_5_20'] = ts_rolling_mean(df['close'], 5) - ts_rolling_mean(df['close'], 20)
        self.df_factors['doubleMA_5_60'] = ts_rolling_mean(df['close'], 5) - ts_rolling_mean(df['close'], 60)
        self.df_factors['doubleMA_10_60'] = ts_rolling_mean(df['close'], 10) - ts_rolling_mean(df['close'], 60)

    def update_factors(self):
        pass