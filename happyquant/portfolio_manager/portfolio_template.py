import pandas as pd
from utils.plot_tools import plot_pnl

def get_list_portfolio(dict_pnl, contracts, chosen_model_id):
    list_portfolio = []
    for name in contracts:
        list_pnl = dict_pnl[name]
        for pnl in list_pnl:
            if pnl.name == f'{name}_{chosen_model_id}':
                list_portfolio.append(pnl)
    return list_portfolio

def get_portfolio_pnl(dict_pnl, contracts, chosen_model_id):
    list_portfolio = get_list_portfolio(dict_pnl, contracts, chosen_model_id)
    df_portfolio = pd.concat(list_portfolio, axis=1)
    df_portfolio.dropna(axis=0, inplace=True)
    df_portfolio['portfolio_equal_weight'] = df_portfolio.mean(axis=1)
    plot_pnl(df_portfolio)