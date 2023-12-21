from datamanager.data_loader import DataLoader
from datamanager.data_recorder import DataRecorder

class DataEngine:
    """
    调用data recorder和data loader的接口
    """
    def load_data(self, folder_path, file_name):
        mydl = DataLoader()
        data = mydl.load_data_by_file(folder_path, file_name)
        print(data)

    def save_data(self, data, folder_path, file_name):
        mydr = DataRecorder(folder_path)
        mydr.save_data_by_file(data, folder_path, file_name)