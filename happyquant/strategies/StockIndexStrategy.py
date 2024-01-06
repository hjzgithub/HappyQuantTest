import os
import abc
import yaml
from typing import Any, Dict

from strategies.StrategyTemplate import StrategyTemplate
from engine.vector_backtest_engine import VectorBacktestEngine

class StockIndexStrategy(StrategyTemplate):
    def __init__(self, **kwargs):
        super(StockIndexStrategy, self).__init__(**kwargs)
        model_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(model_dir, "configs", "StockIndexStrategy.yaml"), 'r') as file:
            params = yaml.safe_load(file)
        self.set_params(**params)
        
    def run_backtest(self, 
                     factor_names = ['TrendFollowing', 'TrendReverse'],
                     contracts = ['000016.SH', '000300.SH', '000905.SH', '000852.SH'],
                     chosen_model_id = 'OLSLRModel',
                     leverage = 1.0,
                     ):
        params = self._params
        myvbe = VectorBacktestEngine(factor_names)
        myvbe.vector_backtest(contracts, **params)
        myvbe.get_portfolio_pnl(chosen_model_id, leverage=leverage)