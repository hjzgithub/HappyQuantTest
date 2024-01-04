import pandas as pd
import importlib

class FactorEngine:
    """
    调用factor_miner挖出的因子
    """
    def __init__(self, factor_name, data_path) -> None:
        factor_file = ".".join(["factor_manager.factor_miner", factor_name])
        factor_module = importlib.import_module(factor_file)
        factor_class = getattr(factor_module, factor_name)
        self.myfm = factor_class(data_path)

    def save_factors(self):
        self.myfm.load_raw_data_from_local()
        self.myfm.init_factors()
        self.myfm.save_factors_to_local()

    def load_factors(self) -> pd.DataFrame:
        self.myfm.load_factors_from_local()
        return self.myfm.df_factors