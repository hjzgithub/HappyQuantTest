from data_manager.data_fetcher import DataFetcher
from engine.data_engine import DataEngine
from engine.factor_engine import FactorEngine
from factor_manager.factor_miner.trend_following import TrendFollowing
from factor_manager.factor_miner.rsi import RSI
from factor_manager.factor_miner.kdj import KDJ

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
    factor_class_list = [TrendFollowing, RSI, KDJ]
    contract = ['000016.SH', '000300.SH', '000905.SH', '000852.SH']
    for factor_class in factor_class_list:
        for name in contract:
            data_path = f'{name[:-3]}.parquet'
            myfe = FactorEngine(factor_class, data_path)
            myfe.save_factors()
            myfe.load_factors()

if __name__ == "__main__":
    #test1()
    #test2()
    test3()