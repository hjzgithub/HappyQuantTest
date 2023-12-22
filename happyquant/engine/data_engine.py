from data_manager.data_loader import DataLoader
from data_manager.data_recorder import DataRecorder

class DataEngine:
    """
    调用data recorder和data loader的接口
    """
    def __init__(self, path_type, store_path):
        self.path_type = path_type
        self.store_path = store_path

    def load_data(self, data_path):
        if self.path_type == 'local':
            mydl = DataLoader()
            return mydl.load_data_by_file(self.store_path, data_path)

    def save_data(self, data, data_path):
        if self.path_type == 'local':
            mydr = DataRecorder(self.store_path)
            mydr.save_data_by_file(data, self.store_path, data_path)