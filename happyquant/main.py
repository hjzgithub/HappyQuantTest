from datamanager.data_fetcher import DataFetcher
from engine.data_engine import DataEngine

def test1():
    mydf = DataFetcher()
    start = '2023-11-20'
    end = '2023-12-20'
    data_source = {'type':'url', 'source':'binance', 'content':'https://data.binance.vision/data'} # 可以写入配置文件
    instrument_type = 'futures'
    update_freq = 'daily'
    data_type = 'klines'
    contract_type = 'BTCDOMUSDT'
    data_freq = '1m'
    data = mydf.fetch_data(start, end, data_source, instrument_type, update_freq, data_type, contract_type, data_freq, use_parallel=True)

    folder_path = f'happyquant/raw_data/crypto/{instrument_type}/{contract_type}'
    file_name = f'{start}_{end}.parquet'
    myde = DataEngine()
    myde.save_data(data, folder_path, file_name)
    myde.load_data(folder_path, file_name)

if __name__ == "__main__":
    test1()