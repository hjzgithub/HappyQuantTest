from abc import ABC, abstractmethod
from engine.data_engine import DataEngine

class FactorManager(ABC):
    def __init__(self, data_path):
        self.data_path = data_path 

    def load_raw_data_from_local(self):
        '''
        从本地获取原始数据
        '''
        self.df_raw = DataEngine('local', self.source_path).load_data(self.data_path)

    def save_factors_to_local(self):
        '''
        因子存储
        '''
        DataEngine('local', self.target_path).save_data(self.df_factors, self.data_path)

    def load_factors_from_local(self):
        '''
        获取现有的因子数据
        '''
        self.df_factors = DataEngine('local', self.target_path).load_data(self.data_path)
    
    @abstractmethod
    def init_factors(self):
        '''
        初始化因子数据
        '''
        pass
    
    @abstractmethod
    def update_factors(self):
        '''
        因子更新
        '''
        pass