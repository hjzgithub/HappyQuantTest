import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

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
    df_cum_rets = (1 + df_rets).cumprod() - 1
    for column in df_cum_rets.columns:
        cum_rets_series = df_cum_rets[column]
        plt.plot(cum_rets_series.index, cum_rets_series.values, label=column)
    plt.legend()
    plt.title('cum rets series')
    plt.show()