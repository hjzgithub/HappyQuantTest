import pandas as pd
from utils.plot_tools import plot_cum_rets, plot_cum_rets_with_excess

def get_portfolio_pnl(list_portfolio_rets, chosen_model_id: str):
    df_portfolio = pd.concat(list_portfolio_rets, axis=1)
    df_portfolio.dropna(axis=0, inplace=True)
    
    chosen_columns = [i for i in df_portfolio.columns if i[-len(chosen_model_id):] == chosen_model_id]
    plot_cum_rets(df_portfolio[chosen_columns])

    df_portfolio['portfolio_equal_weight'] = df_portfolio[chosen_columns].mean(axis=1)
    benchmark_columns = [i for i in df_portfolio.columns if i[-9:] == 'benchmark']
    df_portfolio['portfolio_benchmark'] = df_portfolio[benchmark_columns].mean(axis=1)
    plot_cum_rets_with_excess(df_portfolio['portfolio_equal_weight'], df_portfolio['portfolio_benchmark'])