from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def fixed_pca(x: np.ndarray, n_chosen: int = None, lower_bound: float = 0.9):
    if n_chosen == None:
        origin = PCA(n_components=x.shape[1])
        origin.fit_transform(x)
        n_chosen = np.where(origin.explained_variance_ratio_.cumsum() > lower_bound)[0][0] + 1
        
    pca_components = PCA(n_components=n_chosen).fit_transform(x)
    return pca_components

def rolling_pca(df, back_window, n_chosen):
    df.fillna(method='ffill', axis=0, inplace=True)
    df.dropna(axis=0, inplace=True)

    list_pca_results = [list(fixed_pca(df.iloc[i-back_window:i].values, n_chosen)[-1]) for i in range(back_window, len(df.index))]

    df_pca = pd.DataFrame(list_pca_results, index=df.index[back_window:])
    df_pca.columns = [('PCA_'+str(i)) for i in range(1, len(df_pca.columns)+1)]
    return df_pca