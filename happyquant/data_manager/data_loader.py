import pandas as pd
from loguru import logger

class DataLoader:
    @staticmethod
    def load_data_by_parquet(folder_path, file_name):
        return pd.read_parquet(f'{folder_path}/{file_name}')

    @staticmethod
    def load_data_by_file(folder_path, file_name):
        save_type = file_name.split('.')[-1]
        if save_type == 'parquet':
            file_path = f'{folder_path}/{file_name}'
            data = pd.read_parquet(file_path)
            logger.info(f'data loaded from {file_path}')
            logger.info(data.head().T)
            return data