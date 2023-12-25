import matplotlib.pyplot as plt
import seaborn as sns

def plot_corr(df):
    '''
    绘制相关系数热力图
    '''
    plt.figure(figsize=(16, 12))
    sns.set_style('white')
    sns.heatmap(df.corr(), annot=True)
    plt.title('corr')
    plt.show()

def plot_pnl(pnl_dict):
    plt.figure(figsize=(16, 12))
    for name in pnl_dict:
        pnl = pnl_dict[name]
        plt.plot(range(len(pnl)), pnl, label=name)
    plt.legend()
    plt.title('pnl')
    plt.show()