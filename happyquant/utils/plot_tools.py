import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_corr(df):
    '''
    绘制相关系数热力图
    '''
    if type(df) == list:
        df = pd.concat(df, axis=1)
    df.dropna(axis=0, inplace=True)
    plt.figure(figsize=(16, 12))
    sns.set_style('white')
    sns.heatmap(df.corr(), annot=True)
    plt.title('corr')
    plt.show()