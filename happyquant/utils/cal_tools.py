import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np

def ts_rolling_mean(df: pd.DataFrame, back_window: int):
    return df.rolling(back_window).mean()

def ts_rolling_z_score(df: pd.DataFrame, back_window: int):
    return ((df - df.rolling(back_window).mean()) / df.rolling(back_window).std(ddof=1)).values

def ts_z_score(df: pd.DataFrame | pd.Series):
    return df.apply(lambda x:(x-x.mean())/x.std(ddof=1))

def ts_rank(df: pd.DataFrame):
    return df.apply(lambda x:x.rank(pct=True))

def ts_expanding_rank(df: pd.DataFrame, back_window: int = None):
    if back_window:
        return df.expanding(min_periods=back_window).rank(pct=True)
    else:
        return df.expanding().rank(pct=True)

def ts_corr(x: pd.Series, y: pd.Series):
    correlation_coefficient, p_value = pearsonr(x.to_numpy().reshape(-1), y.to_numpy().reshape(-1))
    return correlation_coefficient, p_value

def ts_rank_corr(x: pd.Series, y: pd.Series):
    correlation_coefficient, p_value = spearmanr(x.to_numpy().reshape(-1), y.to_numpy().reshape(-1))
    return correlation_coefficient, p_value

def ts_rolling_max(x: pd.Series, back_window: int):
    return x.rolling(back_window).max()

def ts_rolling_min(x: pd.Series, back_window: int):
    return x.rolling(back_window).min()

def ts_rolling_ewma(x: pd.Series, back_window: int, recursively=False):
    if recursively:
        return x.ewm(com=back_window - 1, adjust=False).mean()
    else:
        return x.ewm(com=back_window - 1, adjust=True).mean()

def get_divided_by_single_bound(x: pd.Series, bound=0, upper_value=1, lower_value=-1):
    return np.where(x > bound, upper_value, lower_value)

def get_divided_by_two_bounds(x: pd.Series, upper_bound=0.8, lower_bound=0.2, upper_value=1, lower_value=-1, mid_value=0):
    return np.select([x>upper_bound, x<lower_bound, ~((x>upper_bound) & (x<lower_bound))],
                             [upper_value, lower_value, mid_value])











def get_cumrets_from_rets(rets: pd.Series):
    rets.iloc[0] = 0
    cum_rets = (1 + rets).cumprod() - 1
    return cum_rets

def get_stats_result(df):
    statistic_res = df.agg(['mean', 'std', 'skew', 'kurt', 'min', 'max']).T
    statistic_res['mean_per_std'] = (statistic_res['mean']/statistic_res['std'])
    return statistic_res

def get_annualized_rets(rets, days_per_year=252):
    return np.mean(rets) * days_per_year

def get_annualized_vol(rets, frequency='daily', days_per_year=252):
    if frequency == 'daily':
        return np.std(rets, ddof=1) * np.sqrt(days_per_year)
    elif frequency == '30m':
        return np.std(rets, ddof=1) * np.sqrt(days_per_year*8)
    
def get_sharpe_ratio(rets, frequency='daily', days_per_year=252, r_f=0):
    return (get_annualized_rets(rets) - r_f) / get_annualized_vol(rets, frequency, days_per_year)

def get_win_ratio(rets): 
    wins = len(rets[rets > 0])
    losses = len(rets[rets < 0])
    return wins / (wins + losses)

def get_win_per_loss(rets):
    average_win = np.mean(rets[rets > 0])
    average_loss = np.mean(rets[rets < 0])
    return -(average_win / average_loss)

def get_annualized_buy_side_turnover(weights, days_per_year=252):
    if len(weights) >= 2:
        return (np.sum(np.abs(weights[1:] - weights[:-1])) + np.abs(weights[0]) + np.abs(weights[-1])) / (len(weights)+1) / 2 * days_per_year
    
def get_1D_array_from_series(x_series: pd.Series):
    return x_series.to_numpy().reshape(-1)

def get_costs_by_annualized_turnover(turnover, costs_rate):
    return turnover * costs_rate