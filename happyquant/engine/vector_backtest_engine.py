from utils.cal_tools import *
from utils.plot_tools import plot_cum_rets, plot_cum_rets_with_excess
from engine.data_engine import DataEngine
from engine.factor_engine import FactorEngine
from engine.model_engine import ModelEngine
import pandas as pd
from loguru import logger

class VectorBacktestEngine:
    def __init__(self, factor_names) -> None:
        self.factor_names = factor_names
        if not (type(factor_names) == list):
            factor_names = [factor_names]
    
    def init_tags_and_factors(self, data_path):

        path_type = 'local'
        store_path = '/root/HappyQuantTest/happyquant/raw_data/stock_index'
        
        myde = DataEngine(path_type, store_path)
        df_raw = myde.load_data(data_path)
        df_tags = df_raw['trade_date'].to_frame()
        df_tags['tag_raw'] = (df_raw['close'] / df_raw['pre_close'] - 1).shift(-1).fillna(0)
        df_tags.set_index('trade_date', inplace=True)
        df_tags.index = pd.to_datetime(df_tags.index, format='%Y%m%d')

        df_factors_total = pd.DataFrame()
        for factor_name in self.factor_names:
            myfe = FactorEngine(factor_name, data_path)
            df_factors = myfe.load_factors()
            df_factors.set_index('trade_date', inplace=True)
            df_factors.index = pd.to_datetime(df_factors.index, format='%Y%m%d')
            df_factors_total = pd.concat([df_factors_total, df_factors], axis=1)
        return df_tags, df_factors_total

    def vector_backtest(self, contracts, **params):
        if type(contracts) == str:
            contracts = [contracts]

        dict_signal = {}
        list_portfolio_rets_total = []
        for contract_name in contracts:
            list_evaluation = []
            list_portfolio_rets = []

            # Construct the data path based on the contract name
            data_path = f'{contract_name[:-3]}.parquet'
            
            df_tags, df_factors = self.init_tags_and_factors(data_path)

            # Benchmark
            eval_label = f'{contract_name}_benchmark'
            rets = df_tags['tag_raw'].fillna(0)
            weights = 1
            portfolio_rets = get_portfolio_rets_from_weights(rets, weights)
            evaluation = portfolio_evaluation(portfolio_rets, eval_label)
            list_evaluation.append(evaluation)
            list_portfolio_rets.append(pd.Series(portfolio_rets, index=df_tags.index, name=eval_label))

            signal = get_signal_from_factor(params['model_type'], 
                                            params['model_name'],
                                            params['model_id'], 
                                            df_factors, 
                                            df_tags, 
                                            params['back_window'], 
                                            params['target_type'], 
                                            params['standardize_method'], 
                                            params['single_test_method'], 
                                            params['combine_method'], 
                                            params['multi_test_method'],
                                            )
            eval_label = f"{contract_name}_{params['model_id']}"
            evaluation, portfolio_rets = signal_evaluation(eval_label,
                                                            signal,
                                                            df_tags, 
                                                        trade_type=params['trade_type'], 
                                                        )
            list_evaluation.append(evaluation)
            list_portfolio_rets.append(portfolio_rets)
            dict_signal[eval_label] = signal
            
            df_evaluation = pd.concat(list_evaluation)
            logger.info(df_evaluation.T)

            # Plot profit and loss (P&L) for each model
            plot_cum_rets(list_portfolio_rets)

            list_portfolio_rets_total += list_portfolio_rets
    
        self.list_portfolio_rets_total = list_portfolio_rets_total.copy()
    
    def get_portfolio_pnl(self, chosen_model_id: str, leverage: float = 1.0, portfolio_methods = ['equal_weight']):
        df_portfolio = pd.concat(self.list_portfolio_rets_total, axis=1)
        df_portfolio.dropna(axis=0, inplace=True)
        
        chosen_columns = [i for i in df_portfolio.columns if i[10:] == chosen_model_id] # 10 为contract_name_ 的长度
        df_portfolio[chosen_columns] = df_portfolio[chosen_columns] * leverage
        plot_cum_rets(df_portfolio[chosen_columns])

        list_evaluation = []

        benchmark_columns = [i for i in df_portfolio.columns if i[-9:] == 'benchmark']
        df_portfolio['portfolio_benchmark'] = df_portfolio[benchmark_columns].mean(axis=1)
        eval_label = f'portfolio_benchmark'
        evaluation = portfolio_evaluation(df_portfolio[eval_label], eval_label)
        list_evaluation.append(evaluation)

        for portfolio_method in portfolio_methods:
            if portfolio_method == 'equal_weight':
                df_portfolio['portfolio_equal_weight'] = df_portfolio[chosen_columns].mean(axis=1) 
    
            eval_label = f'portfolio_{portfolio_method}'
            evaluation = portfolio_evaluation(df_portfolio[eval_label], eval_label)
            list_evaluation.append(evaluation)

        df_evaluation = pd.concat(list_evaluation)
        logger.info(df_evaluation.T)
        plot_cum_rets_with_excess(df_portfolio[eval_label], df_portfolio['portfolio_benchmark'])

def get_signal_from_factor(model_type: str, 
                           model_name: str, 
                           model_id: str,
                           df_factors: pd.DataFrame, 
                           df_tags: pd.DataFrame, 
                           back_window: int, 
                           target_type, 
                           standardize_method, 
                           single_test_method, 
                           combine_method, 
                           multi_test_method,
                           ):
    # 注：signal是需要归一化的
    if model_type == 'rule_based':
        if model_name == 'vote':
            # 因子生成离散信号，再等权合成 -- 弊端：比较依赖先验知识
            signal = df_factors.mean(axis=1) 

    elif model_type == 'prediction_based':
        # 机器学习预测
        myme = ModelEngine()
        df_preds = myme.rolling_run_models(model_name, model_id, df_factors, df_tags['tag_raw'], back_window, target_type, \
                                      standardize_method, single_test_method, combine_method, multi_test_method)

        if target_type == 'tag_raw':
            signal = pd.Series(get_divided_by_single_bound(df_preds).reshape(-1), index=df_preds.index)
        elif target_type == 'tag_ranked':
            signal = pd.Series(get_divided_by_single_bound(df_preds, 0.5).reshape(-1), index=df_preds.index)
        elif (target_type == 'tag_class') | (target_type == 'tag_multi_class'):
            signal = df_preds.copy()             
    return signal
    
def get_weights_from_signal(signal: pd.Series, trade_type: str) -> np.ndarray:
    signal = get_1D_array_from_series(signal)
    if trade_type == 'long_only':
        weights = (signal + 1) / 2
    elif trade_type == 'long_short':
        weights = signal
    weights = get_1D_array_from_series(pd.Series(weights).fillna(method='ffill').fillna(0))
    return weights

def get_portfolio_rets_from_weights(rets, weights):
    portfolio_rets = get_1D_array_from_series(rets) * weights
    return portfolio_rets

def signal_evaluation(eval_label: str,
                      signal: pd.Series,
                      df_tags: pd.DataFrame, 
                      trade_type: str, 
                      ):
    rets = df_tags['tag_raw'].loc[signal.index].fillna(0)
    weights = get_weights_from_signal(signal, trade_type)
    annualized_turnover = get_annualized_buy_side_turnover(weights)
    portfolio_rets = get_portfolio_rets_from_weights(rets, weights)

    evaluation = pd.Series({
                        'annualized turnover(buy side)': annualized_turnover,
                        'annualized return': get_annualized_rets(portfolio_rets),
                        'annualized costs': get_costs_by_annualized_turnover(annualized_turnover, costs_rate=0.00023*2),
                        'sharpe ratio': get_sharpe_ratio(portfolio_rets),
                        'win ratio': get_win_ratio(portfolio_rets),
                        'win per loss': get_win_per_loss(portfolio_rets),
                        }, name=eval_label).to_frame().T

    portfolio_rets_series = pd.Series(portfolio_rets, index=signal.index, name=eval_label)
    return evaluation, portfolio_rets_series

def portfolio_evaluation(portfolio_rets: np.ndarray, eval_label: str):
    evaluation = pd.Series({
                            'annualized return': get_annualized_rets(portfolio_rets),
                            'sharpe ratio': get_sharpe_ratio(portfolio_rets),
                            'win ratio': get_win_ratio(portfolio_rets),
                            'win per loss': get_win_per_loss(portfolio_rets),
                            }, name=eval_label).to_frame().T
    return evaluation