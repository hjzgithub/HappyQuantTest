import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from factor_manager.factor_template import FactorManager
from utils.cal_tools import ts_rolling_mean

class RSI(FactorManager):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.source_path = '/root/HappyQuantTest/happyquant/raw_data/stock_index'
        self.target_path = '/root/HappyQuantTest/happyquant/factors/stock_index/rsi'

    def init_factors(
            self, 
            window_list=[9, 14, 21, 28],
        ):
        df = self.df_raw[['trade_date', 'close', 'pre_close']].copy()
        df['daily gain'] = df['close'] - df['pre_close']
        gain_series = df['daily gain'].apply(lambda x: x if x>0 else 0)
        loss_series = df['daily gain'].apply(lambda x: -x if x<0 else 0)

        self.df_factors = df['trade_date'].to_frame()
        for n in window_list:  
            self.df_factors['minus_rsi_' + str(n)] = -cal_rsi(gain_series, loss_series, n) # 取负号代表为超买超卖信号

    def update_factors(self):
        pass

def cal_rsi(gain_series, loss_series, n):
    relative_strength = ts_rolling_mean(gain_series, n) / ts_rolling_mean(loss_series, n)
    return relative_strength / (1 + relative_strength)