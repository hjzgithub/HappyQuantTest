from utils.cal_tools import *

def get_signal_from_factor(factor, method):
    if method == 'expanding_rank':
        signal = factor.expanding().rank(pct=True)
    return signal
    
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

# 仓位到组合收益
def get_portfolio_rets(rets, weights):
    return get_1D_array_from_series(rets) * weights

# 对组合收益率的评估
def benchmark_evaluation(df_tags, eval_label):
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

def model_evaluation(df_tags, df_preds, eval_label, method, trade_type='long short', upper_bound=0, lower_bound=0):
    rets = df_tags['tag_raw'].loc[df_preds.index].fillna(0)
    singal = get_signal_from_factor(df_preds, method)
    weights = get_weights_from_signal(singal, trade_type, upper_bound, lower_bound)
    
    turnover = get_buy_side_turnover(weights)
    portfolio_rets = get_portfolio_rets(rets, weights)

    evaluation = pd.Series({
                        'turnover(buy side)': turnover,
                        'annualized return': get_annualized_rets(portfolio_rets),
                        'annualized costs': get_annualized_costs_by_turnover(turnover, costs_rate=0.00023),
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