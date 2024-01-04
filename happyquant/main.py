from data_manager.data_fetcher import DataFetcher
from engine.data_engine import DataEngine
from engine.factor_engine import FactorEngine
import argparse

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
    factor_class_list = ['TrendFollowing', 'TrendFollowingDiscretized', 'TrendReverse', 'TrendReverseDiscretized']
    contract = ['000016.SH', '000300.SH', '000905.SH', '000852.SH']
    for factor_class in factor_class_list:
        for name in contract:
            data_path = f'{name[:-3]}.parquet'
            myfe = FactorEngine(factor_class, data_path)
            myfe.save_factors()
            myfe.load_factors()

def get_args_with_batch_id(batch_id: int) -> argparse.Namespace:
    """
    Get command line arguments with default values based on the provided batch_id.

    Parameters:
        batch_id (int): Identifier for selecting a specific set of default arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor_name", "-m", type=str, default="TrendFollowing", help="TrendFollowing factor")
    parser.add_argument("--model_type", type=str, default="rule_based", help="Model type")
    parser.add_argument("--model_name", type=str, default="vote", help="Model name")
    parser.add_argument("--back_window", type=int, default=3000, help="Number of trading days for train data")
    parser.add_argument("--target_type", type=str, default="tag_raw", help="tag_raw or tag_ranked in model prediction")
    parser.add_argument("--with_pca", type=bool, default=False, help="PCA in model training")
    parser.add_argument("--trade_type", type=str, default='long_short', help="Long only or long short in signal evaluation")
    parser.add_argument("--upper_bound", type=float, default=0, help="Upper bound of signal to weight")
    parser.add_argument("--lower_bound", type=float, default=0, help="Lower bound of signal to weight")
    parser.add_argument("--model_id", type=str, default="vote", help="Model id created from Model name")
    args = parser.parse_args()
    
    if batch_id == 1:
        args.model_type = 'prediction_based'
        args.model_name = 'OLSLRModel'
        args.target_type = 'tag_raw'   
        args.with_pca = False
        args.model_id = 'OLSLRModel'
    
    elif batch_id == 2:
        args.model_type = 'prediction_based'
        args.model_name = 'OLSLRModel'
        args.target_type = 'tag_ranked'   
        args.with_pca = False
        args.model_id = 'OLSLRModel_tag_ranked'

    elif batch_id == 3:
        args.model_type = 'prediction_based'
        args.model_name = 'OLSLRModel'
        args.target_type = 'tag_ranked'   
        args.with_pca = True
        args.model_id = 'OLSLRModel_tag_ranked_PCA'

    elif batch_id == 4:
        args.model_type = 'prediction_based'
        args.model_name = 'StatsOLSLRModel'
        args.target_type = 'tag_ranked'   
        args.with_pca = False
        args.model_id = 'StatsOLSLRModel_tag_ranked'

    elif batch_id == 5:
        args.model_type = 'prediction_based'
        args.model_name = 'LassoLRModel'
        args.target_type = 'tag_ranked'   
        args.with_pca = False
        args.model_id = 'LassoLRModel_tag_ranked'

    elif batch_id == 10:
        args.model_type = 'prediction_based'
        args.model_name = 'LogisticModel'
        args.target_type = 'tag_class'   
        args.with_pca = False
        args.model_id = 'LogisticModel_tag_class'

    elif batch_id == 11:
        args.model_type = 'prediction_based'
        args.model_name = 'LogisticModel'
        args.target_type = 'tag_multi_class'   
        args.with_pca = False
        args.model_id = 'LogisticModel_tag_multi_class'

    return args

if __name__ == "__main__":
    #test1()
    #test2()
    test3()