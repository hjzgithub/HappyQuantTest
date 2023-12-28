import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from factor_manager.factor_template import FactorManager
from utils.cal_tools import ts_rolling_mean, ts_rolling_ewma

class RUMI(FactorManager):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.source_path = '/root/HappyQuantTest/happyquant/raw_data/stock_index'
        self.target_path = '/root/HappyQuantTest/happyquant/factors/stock_index/rumi'

    def init_factors(
            self, 
            columns=['trade_date', 'close'],
        ):
        df = self.df_raw[columns].copy()
        self.df_factors = df['trade_date'].to_frame()
        self.df_factors['rumi_5_20_10'] = cal_rumi(df['close'], 5, 20, 10)
        self.df_factors['rumi_5_60_10'] = cal_rumi(df['close'], 5, 60, 10)
        self.df_factors['rumi_5_60_20'] = cal_rumi(df['close'], 5, 60, 20)

    def update_factors(self):
        pass

def cal_rumi(close_series, fast_window, slow_window, rumi_window):
    fast = ts_rolling_mean(close_series, fast_window)
    slow = ts_rolling_ewma(close_series, slow_window)
    return ts_rolling_mean(fast-slow, rumi_window)

# https://blog.csdn.net/weixin_59313523/article/details/134432961