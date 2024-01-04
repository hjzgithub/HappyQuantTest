from utils.cal_tools import *
from utils.plot_tools import plot_corr, plot_cum_rets, plot_cum_rets_with_excess
from engine.data_engine import DataEngine
from engine.factor_engine import FactorEngine
from portfolio_manager.run_models import rolling_run_models
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
        df_tags['tag_class'] = get_divided_by_single_bound(df_tags['tag_raw'])
        df_tags['tag_ranked'] = ts_expanding_rank(df_tags['tag_raw'])
        df_tags['tag_multi_class'] = get_divided_by_two_bounds(df_tags['tag_ranked'], 0.7, 0.3)
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

    def vector_backtest(self, contracts, args_list):
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

            # Plot correlation between tags and factors
            plot_corr([df_tags, df_factors])

            # Benchmark
            eval_label = f'{contract_name}_benchmark'
            rets = df_tags['tag_raw'].fillna(0)
            weights = 1
            portfolio_rets = get_portfolio_rets_from_weights(rets, weights)
            evaluation = portfolio_evaluation(portfolio_rets, eval_label)
            list_evaluation.append(evaluation)
            list_portfolio_rets.append(pd.Series(portfolio_rets, index=df_tags.index, name=eval_label))

            # Args Test
            for new_args in args_list:
                signal = get_signal_from_factor(new_args.model_type, 
                                                new_args.model_name,
                                                new_args.model_id, 
                                                new_args.target_type,
                                                df_factors, 
                                                df_tags, 
                                                new_args.back_window, 
                                                new_args.with_ts_z_score,
                                                new_args.with_pca)
                eval_label = f'{contract_name}_{new_args.model_id}'
                evaluation, portfolio_rets = signal_evaluation(eval_label,
                                                                signal,
                                                                df_tags, 
                                                            trade_type=new_args.trade_type, 
                                                            upper_bound=new_args.upper_bound, 
                                                            lower_bound=new_args.lower_bound)
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
        
        chosen_columns = [i for i in df_portfolio.columns if i[-len(chosen_model_id):] == chosen_model_id]
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
                           target_type: str, 
                           df_factors: pd.DataFrame, 
                           df_tags: pd.DataFrame, 
                           back_window: int = None, 
                           with_ts_z_score: bool = False,
                           with_pca: bool = False,
                           ):
    # 注：signal是需要归一化的
    if model_type == 'rule_based':
        if model_name == 'vote':
            # 因子生成离散信号，再等权合成 -- 弊端：比较依赖先验知识
            signal = df_factors.mean(axis=1) 

    elif model_type == 'prediction_based':
        # 因子合成单机器学习因子，预测目标为tag_raw或者tag_class
        df_preds = rolling_run_models(model_name, model_id, df_factors, df_tags[target_type], back_window, with_ts_z_score, with_pca)

        if target_type == 'tag_raw':
            signal = pd.Series(get_divided_by_single_bound(df_preds).reshape(-1), index=df_preds.index)
        elif target_type == 'tag_ranked':
            signal = pd.Series(get_divided_by_single_bound(df_preds, 0.5).reshape(-1), index=df_preds.index)
        elif (target_type == 'tag_class') | (target_type == 'tag_multi_class'):
            signal = df_preds.copy()             
    return signal
    
def get_weights_from_signal(signal: pd.Series, trade_type: str, upper_bound: float, lower_bound: float) -> np.ndarray:
    """
    Generate weights based on a trading signal.

    Parameters:
    - signal (pd.Series): The trading signal.
    - trade_type (str): Type of trading strategy ('long_only' or 'long_short').
    - upper_bound (float): Upper threshold for the trading signal.
    - lower_bound (float): Lower threshold for the trading signal.

    Returns:
    - np.ndarray: Array of weights generated based on the signal and strategy.
    """
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
                      upper_bound: float, 
                      lower_bound: float,
                      ):
    rets = df_tags['tag_raw'].loc[signal.index].fillna(0)
    weights = get_weights_from_signal(signal, trade_type, upper_bound, lower_bound)
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