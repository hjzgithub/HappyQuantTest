import os

class DataRecorder:
    def __init__(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)

    @staticmethod
    def save_data_by_parquet(data, folder_path, file_name):
        return data.to_parquet(f'{folder_path}/{file_name}')
    
    @staticmethod
    def save_data_by_file(data, folder_path, file_name):
        save_type = file_name.split('.')[-1]
        if save_type == 'parquet':
            data.to_parquet(f'{folder_path}/{file_name}')