import pandas as pd

class FactorEngine:
    """
    调用factor_miner挖出的因子
    """
    def __init__(self, factor_class, data_path) -> None:
        self.myfm = factor_class(data_path)

    def save_factors(self):
        self.myfm.load_raw_data_from_local()
        self.myfm.init_factors()
        self.myfm.save_factors_to_local()

    def load_factors(self) -> pd.DataFrame:
        self.myfm.load_factors_from_local()
        return self.myfm.df_factors