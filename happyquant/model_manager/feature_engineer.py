from utils.cal_tools import *
from model_manager.Handler import Handler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def trans_tag(y, method):
    if method == 'tag_class':
        y = get_divided_by_single_bound(y)
    return y

def standardize_feature(X, method):
    ''' 
    这里对于不同的feature应该采取不同的标准化方法
    '''
    if method == 'ts_z_score':
        X = ts_z_score(pd.DataFrame(X)).to_numpy()
    elif method == 'ts_rank':
        X = ts_rank(pd.DataFrame(X)).to_numpy()
    return X

def single_factor_test(X, y, method):
    ''' 
    单因子测试
    '''
    X_train, X_test = X[:-1, :], X[-1, :].reshape(1, -1)
    y_train, y_test = y[:-1], y[-1] 
    if method == 'ic':
        selected_indices = np.where(abs(np.array([ts_corr(X_train[:, i], y_train)[0] for i in range(X_train.shape[1])])) > 0.05)[0]
        if len(selected_indices) > 0:
            X = X[:, np.ix_(selected_indices)[0]]
    return X

def combine_feature(X, y, method):
    ''' 
    因子合并
    '''
    if method == 'pca':
        X = fixed_pca(X, lower_bound=0.95)
    return X

def multi_factor_test(X, y, method):
    '''
    多因子测试
    '''
    X_train, X_test = X[:-1, :], X[-1, :].reshape(1, -1)
    y_train, y_test = y[:-1], y[-1] 
    handler = Handler()
    model_id, model = handler.new_model(model_name=method, model_id=f'{method}_fe')
    handler.train_model(model_id, X_train, y_train)
    if method == 'StatsOLSLRModel':
        selected_indices = np.where(handler._models[model_id]._model.pvalues[1:] < 0.1)[0]
        if len(selected_indices) > 0:
            X = X[:, np.ix_(selected_indices)[0]]
    return X
        
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