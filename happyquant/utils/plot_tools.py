import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.cal_tools import get_cumrets_from_rets

def plot_corr(df):
    '''
    绘制相关系数热力图
    '''
    if type(df) == list:
        df = pd.concat(df, axis=1)
        df.dropna(axis=0, inplace=True)

    plt.figure(figsize=(12, 8))
    sns.set_style('white')
    sns.heatmap(df.corr(), annot=True)
    plt.title('corr')
    plt.show()

def plot_cum_rets(df_rets):
    if type(df_rets) == list:
        df_rets = pd.concat(df_rets, axis=1)
        df_rets.dropna(axis=0, inplace=True)
        
    plt.figure(figsize=(12, 8))
    df_cum_rets = df_rets.apply(get_cumrets_from_rets)
    for column in df_cum_rets.columns:
        cum_rets_series = df_cum_rets[column]
        plt.plot(cum_rets_series.index, cum_rets_series.values, label=column)
    plt.legend()
    plt.title('cum rets series')
    plt.show()

def plot_cum_rets_with_excess(df_rets, df_benchmark): 
    plt.figure(figsize=(12, 8))
    
    cum_rets_series = get_cumrets_from_rets(df_rets)
    benchmark_series = get_cumrets_from_rets(df_benchmark.loc[cum_rets_series.index])
    excess_series = (1+cum_rets_series)/(1+benchmark_series) - 1
    
    plt.plot(benchmark_series.index, benchmark_series.values, label='benchmark')
    plt.plot(cum_rets_series.index, cum_rets_series.values, label='portfolio')
    plt.plot(excess_series.index, excess_series.values, label='excess', color='gray')

    plt.legend()
    plt.title('cum rets series with excess')
    plt.show()