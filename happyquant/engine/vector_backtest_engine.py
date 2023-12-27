from utils.cal_tools import *
from utils.plot_tools import plot_corr, plot_pnl
from models.pca import rolling_pca
from models.run_models import rolling_run_models
import pandas as pd
from loguru import logger
    
def get_weights_from_signal(signal, trade_type, upper_bound, lower_bound):
    signal = get_1D_array_from_series(signal)
    if trade_type == 'long_only':
        weights = np.select([signal>upper_bound, signal<lower_bound, ~((signal>upper_bound) & (signal<lower_bound))],
                             [1, 0, np.nan])
        weights = get_1D_array_from_series(pd.Series(weights).fillna(method='ffill').fillna(0))
    elif trade_type == 'long_short':
        weights = np.select([signal>upper_bound, signal<lower_bound, ~((signal>upper_bound) & (signal<lower_bound))], 
                            [1, -1, 0])
    return weights


def get_portfolio_rets(rets, weights):
    '''
    仓位到组合收益
    '''
    return get_1D_array_from_series(rets) * weights

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
    
    pnl = pd.Series((1 + portfolio_rets).cumprod() - 1, name=eval_label)
    return evaluation, pnl

def init_tags_and_factors(mymm):
    # Load raw data from the local storage
    mymm.load_raw_data_from_local()

    # Create a DataFrame for tags based on the raw data
    df_tags = mymm.df_raw['trade_date'].to_frame()
    df_tags['tag_raw'] = (mymm.df_raw['close'] / mymm.df_raw['pre_close'] - 1).shift(-1).fillna(0)
    df_tags['tag_ranked'] = df_tags['tag_raw'].expanding().rank(pct=True)
    df_tags.set_index('trade_date', inplace=True)

    # Load factors from local storage
    mymm.load_factors_from_local()
    df_factors = mymm.df_factors.copy()
    df_factors.set_index('trade_date', inplace=True)
    return df_tags, df_factors

def get_signal_from_factor(df_factors, signal_method, df_tags=None, back_window=None, with_pca=None):
    if signal_method == 'equal_weight':
        signal = df_factors.fillna(0).expanding().rank(pct=True).mean(axis=1) # 因子生成信号,再等权合成 -- 弊端：无法判断因子的方向
    elif signal_method == 'prediction':
        df_preds = rolling_run_models(df_factors, df_tags, back_window, with_pca) # 因子合成单机器学习因子，预测目标为tag_raw或者tag_ranked
        signal = df_preds.expanding().rank(pct=True)
    return signal

def signal_evaluation(eval_label,
                    signal,
                    df_tags, 
                    trade_type='long short', 
                    upper_bound=0, 
                    lower_bound=0):
    rets = df_tags['tag_raw'].loc[signal.index].fillna(0)
    weights = get_weights_from_signal(signal, trade_type, upper_bound, lower_bound)
    turnover = get_buy_side_turnover(weights)
    portfolio_rets = get_portfolio_rets(rets, weights)

    evaluation = pd.Series({
                        'turnover(buy side)': turnover,
                        'annualized return': get_annualized_rets(portfolio_rets),
                        'annualized costs': get_annualized_costs_by_turnover(turnover, costs_rate=0.00023*2),
                        'sharpe ratio': get_sharpe_ratio(portfolio_rets),
                        'win ratio': get_win_ratio(portfolio_rets),
                        'win per loss': get_win_per_loss(portfolio_rets),
                        }, name=eval_label).to_frame().T

    pnl = pd.Series((1 + portfolio_rets).cumprod() - 1, name=eval_label)
    return evaluation, pnl

def vector_backtest(factor_class, contracts, args_list):
    if type(contracts) == str:
        contracts = [contracts]

    for name in contracts:
        list_evaluation = []
        list_pnl = []

        # Construct the data path based on the contract name
        data_path = f'{name[:-3]}.parquet'

        # Create an instance of the Momentum class
        mymm = factor_class(data_path)
        df_tags, df_factors = init_tags_and_factors(mymm)

        # Plot correlation between tags and factors
        plot_corr([df_tags, df_factors])

        # Benchmark
        evaluation, pnl = benchmark_evaluation(df_tags, f'{name}_benchmark')
        list_evaluation.append(evaluation)
        list_pnl.append(pnl)

        # Args Test
        for new_args in args_list:
            signal = get_signal_from_factor(df_factors, new_args.signal_method, df_tags=df_tags[new_args.target], back_window=new_args.back_window, with_pca=new_args.with_pca)
            evaluation, pnl = signal_evaluation(f'{name}_{new_args.model_id}',
                                                signal,
                                                df_tags, 
                                            trade_type=new_args.trade_type, 
                                            upper_bound=new_args.upper_bound, 
                                            lower_bound=new_args.lower_bound)
            list_evaluation.append(evaluation)
            list_pnl.append(pnl)
        
        df_evaluation = pd.concat(list_evaluation)
        logger.info(df_evaluation.T)

        # Plot profit and loss (P&L) for each model
        plot_pnl(list_pnl)

    if len(contracts) == 1 and len(args_list) == 1:
        return signal
    
def vector_backtest_from_factors(factor_class, contracts, args_list, df_factors):
    if type(contracts) == str:
        contracts = [contracts]

    for name in contracts:
        list_evaluation = []
        list_pnl = []

        # Construct the data path based on the contract name
        data_path = f'{name[:-3]}.parquet'

        # Create an instance of the Momentum class
        mymm = factor_class(data_path)
        df_tags, _ = init_tags_and_factors(mymm)

        # Plot correlation between tags and factors
        plot_corr([df_tags, df_factors])

        # Benchmark
        evaluation, pnl = benchmark_evaluation(df_tags, f'{name}_benchmark')
        list_evaluation.append(evaluation)
        list_pnl.append(pnl)

        # Args Test
        for new_args in args_list:
            signal = get_signal_from_factor(df_factors, new_args.signal_method, df_tags=df_tags[new_args.target], back_window=new_args.back_window, with_pca=new_args.with_pca)
            evaluation, pnl = signal_evaluation(f'{name}_{new_args.model_id}',
                                                signal,
                                                df_tags, 
                                            trade_type=new_args.trade_type, 
                                            upper_bound=new_args.upper_bound, 
                                            lower_bound=new_args.lower_bound)
            list_evaluation.append(evaluation)
            list_pnl.append(pnl)
        
        df_evaluation = pd.concat(list_evaluation)
        logger.info(df_evaluation.T)

        # Plot profit and loss (P&L) for each model
        plot_pnl(list_pnl)