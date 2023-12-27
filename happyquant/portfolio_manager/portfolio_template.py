import pandas as pd
from utils.plot_tools import plot_pnl

def get_portfolio_pnl(portfolio_pnl_list):
    df_portfolio = pd.concat(portfolio_pnl_list, axis=1)
    df_portfolio.dropna(axis=0, inplace=True)
    df_portfolio['equal_weight'] = df_portfolio.mean(axis=1)
    plot_pnl(df_portfolio)