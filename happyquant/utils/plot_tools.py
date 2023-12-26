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

def plot_pnl(list_df: List):
    plt.figure(figsize=(12, 8))
    for pnl in list_df:
        plt.plot(range(len(pnl)), pnl, label=pnl.name)
    plt.legend()
    plt.title('pnl')
    plt.show()