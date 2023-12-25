from data_manager.data_fetcher import DataFetcher
from engine.data_engine import DataEngine
from factor_manager.momentum import Momentum
from factor_manager.doubleMA import DoubleMA

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
    use_parallel = True
    data = mydf.fetch_data(start, end, data_source, instrument_type, update_freq, data_type, contract_type, data_freq, use_parallel)

    store_path = f'happyquant/raw_data/crypto/{instrument_type}/{contract_type}'
    data_path = f'{start}_{end}.parquet'
    myde = DataEngine('local', store_path)
    myde.save_data(data, data_path)
    myde.load_data(data_path)

def test2():
    mydf = DataFetcher()
    start = None
    end = None
    data_source = {'type':'api', 'source':'tushare'} # 可以写入配置文件
    instrument_type = 'stock_index'
    update_freq = 'total'
    data_type = 'klines'
    contract = ['000016.SH', '000300.SH', '000905.SH', '000852.SH']
    data_freq = '1d'
    use_parallel = True
    data = mydf.fetch_data(start, end, data_source, instrument_type, update_freq, data_type, contract, data_freq, use_parallel)
    
    store_path = f'happyquant/raw_data/{instrument_type}'
    myde = DataEngine('local', store_path)
    for name, df in data.groupby('ts_code'):
        df.reset_index(drop=True, inplace=True)
        data_path = f'{name[:-3]}.parquet'
        myde.save_data(df, data_path)
        myde.load_data(data_path)

def test3():
    contract = ['000016.SH', '000300.SH', '000905.SH', '000852.SH']
    for name in contract:
        data_path = f'{name[:-3]}.parquet'
        mymm = Momentum(data_path)
        mymm.load_raw_data_from_local()
        mymm.init_factors()
        mymm.save_factors_to_local()
        mymm.load_factors_from_local()

def test4():
    contract = ['000016.SH', '000300.SH', '000905.SH', '000852.SH']
    for name in contract:
        data_path = f'{name[:-3]}.parquet'
        mymm = DoubleMA(data_path)
        mymm.load_raw_data_from_local()
        mymm.init_factors()
        mymm.save_factors_to_local()
        mymm.load_factors_from_local()

if __name__ == "__main__":
    #test1()
    #test2()
    test3()
    test4()