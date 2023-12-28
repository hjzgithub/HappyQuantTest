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

def plot_pnl(df):
    if type(df) == list:
        df = pd.concat(df, axis=1)
        df.dropna(axis=0, inplace=True)
    plt.figure(figsize=(12, 8))
    for column in df.columns:
        pnl = df[column]
        plt.plot(pnl.index, pnl.values, label=column)
    plt.legend()
    plt.title('pnl')
    plt.show()