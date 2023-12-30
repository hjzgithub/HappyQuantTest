from utils.cal_tools import *
from utils.plot_tools import plot_corr, plot_cum_rets
from models.run_models import rolling_run_models
import pandas as pd
from loguru import logger
    
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
        weights = np.select([signal>upper_bound, signal<lower_bound, ~((signal>upper_bound) & (signal<lower_bound))],
                             [1, 0, 0])
    elif trade_type == 'long_short':
        weights = np.select([signal>upper_bound, signal<lower_bound, ~((signal>upper_bound) & (signal<lower_bound))], 
                            [1, -1, 0])
    weights = get_1D_array_from_series(pd.Series(weights).fillna(method='ffill').fillna(0))
    return weights

def get_portfolio_rets(rets, weights):
    '''
    仓位到组合收益
    '''
    return get_1D_array_from_series(rets) * weights

def init_tags_and_factors(factor_classes, data_path):
    if not (type(factor_classes) == list):
        factor_classes = [factor_classes]
    k = 0
    df_factors = pd.DataFrame()
    for factor_class in factor_classes:
        # Create an instance of the Momentum class
        mymm = factor_class(data_path)

        if k == 0:
            # Load raw data from the local storage
            mymm.load_raw_data_from_local()

            # Create a DataFrame for tags based on the raw data
            df_tags = mymm.df_raw['trade_date'].to_frame()
            df_tags['tag_raw'] = (mymm.df_raw['close'] / mymm.df_raw['pre_close'] - 1).shift(-1).fillna(0)
            df_tags['tag_class'] = np.where(df_tags['tag_raw']>0, 1, 0)
            df_tags.set_index('trade_date', inplace=True)
            df_tags.index = pd.to_datetime(df_tags.index, format='%Y%m%d')
            k = 1

        # Load factors from local storage
        mymm.load_factors_from_local()
        mymm.df_factors.set_index('trade_date', inplace=True)
        mymm.df_factors.index = pd.to_datetime(mymm.df_factors.index, format='%Y%m%d')
        df_factors = pd.concat([df_factors, mymm.df_factors], axis=1)

    return df_tags, df_factors

def benchmark_evaluation(df_tags, eval_label):
    '''
    对组合收益率的评估
    '''
    rets = df_tags['tag_raw'].fillna(0)
    weights = 1
    portfolio_rets = get_portfolio_rets(rets, weights)
    
    evaluation = pd.Series({
                            'annualized return': get_annualized_rets(portfolio_rets),
                            'sharpe ratio': get_sharpe_ratio(portfolio_rets),
                            'win ratio': get_win_ratio(portfolio_rets),
                            'win per loss': get_win_per_loss(portfolio_rets),
                            }, name=eval_label).to_frame().T
    
    portfolio_rets_series = pd.Series(portfolio_rets, index=df_tags.index, name=eval_label)
    return evaluation, portfolio_rets_series

def get_signal_from_factor(model_name, df_factors: pd.DataFrame, signal_method, df_tags=None, back_window=None, with_pca=None):
    # 注：signal是需要归一化的
    if signal_method == 'equal_weight':
        # 因子生成信号,再等权合成 -- 弊端：无法判断因子的方向
        signal = df_factors.fillna(0).expanding(min_periods=back_window).rank(pct=True).fillna(0.5).mean(axis=1) 
    elif signal_method == 'regression':
        # 因子合成单机器学习因子，预测目标为tag_raw或者tag_ranked 
        df_preds = rolling_run_models(model_name, df_factors, df_tags, back_window, with_pca)
        signal = df_preds.expanding().rank(pct=True)
    elif signal_method == 'classification':
        df_preds = rolling_run_models(model_name, df_factors, df_tags, back_window, with_pca)
        signal = df_preds.copy()
    return signal

def signal_evaluation(eval_label,
                    signal,
                    df_tags, 
                    trade_type, 
                    upper_bound, 
                    lower_bound):
    rets = df_tags['tag_raw'].loc[signal.index].fillna(0)
    weights = get_weights_from_signal(signal, trade_type, upper_bound, lower_bound)
    annualized_turnover = get_annualized_buy_side_turnover(weights)
    portfolio_rets = get_portfolio_rets(rets, weights)

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

def vector_backtest(factor_classes, contracts, args_list):
    if type(contracts) == str:
        contracts = [contracts]

    dict_signal = {}
    dict_portfolio_rets = {}
    for contract_name in contracts:
        list_evaluation = []
        list_portfolio_rets = []

        # Construct the data path based on the contract name
        data_path = f'{contract_name[:-3]}.parquet'
        
        df_tags, df_factors = init_tags_and_factors(factor_classes, data_path)

        # Plot correlation between tags and factors
        plot_corr([df_tags, df_factors])

        # Benchmark
        evaluation, portfolio_rets = benchmark_evaluation(df_tags, f'{contract_name}_benchmark')
        list_evaluation.append(evaluation)
        list_portfolio_rets.append(portfolio_rets)

        # Args Test
        for new_args in args_list:
            signal = get_signal_from_factor(new_args.model_name, df_factors, new_args.signal_method, df_tags=df_tags[new_args.target], back_window=new_args.back_window, with_pca=new_args.with_pca)
            evaluation, portfolio_rets = signal_evaluation(f'{contract_name}_{new_args.model_id}',
                                                signal,
                                                df_tags, 
                                            trade_type=new_args.trade_type, 
                                            upper_bound=new_args.upper_bound, 
                                            lower_bound=new_args.lower_bound)
            list_evaluation.append(evaluation)
            list_portfolio_rets.append(portfolio_rets)
            dict_signal[f'{contract_name}_{new_args.model_id}'] = signal
        
        df_evaluation = pd.concat(list_evaluation)
        logger.info(df_evaluation.T)

        # Plot profit and loss (P&L) for each model
        plot_cum_rets(list_portfolio_rets)

        dict_portfolio_rets[contract_name] = list_portfolio_rets
 
    return dict_signal, dict_portfolio_rets