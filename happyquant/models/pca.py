from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def fixed_pca(df_standard, n_chosen):
    x = df_standard.values

    n_total = len(df_standard.columns)
    origin = PCA(n_components=n_total)
    origin_components = origin.fit_transform(x)
    importance = origin.explained_variance_ratio_
    n_chosen = min(np.where(importance.cumsum() > 0.9)[0][0] + 1, n_chosen)

    pca = PCA(n_components=n_chosen)
    pca_components = pca.fit_transform(x)

    df_pca = pd.DataFrame(pca_components, columns=[('PCA_'+str(i)) for i in range(1, n_chosen+1)], index=df_standard.index)
    return df_pca

def rolling_pca(df, back_window, n_chosen=2):
    df.fillna(method='ffill', axis=0, inplace=True)
    df.dropna(axis=0, inplace=True)
    df_pca = pd.DataFrame(index=df.index, columns=[('PCA_'+str(i)) for i in range(1, n_chosen+1)])
    for i in range(back_window, len(df.index)):
        pca_results = fixed_pca(df.iloc[i-back_window:i], n_chosen).iloc[-1].values
        df_pca.iloc[i] = pca_results
    return df_pca