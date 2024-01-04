from portfolio_manager.Handler import Handler
from utils.pca_tools import fixed_pca
from utils.cal_tools import ts_z_score
import pandas as pd
import joblib 
import numpy as np

def get_data_split(df_factors, df_tags, i, back_window, with_ts_z_score, with_pca):
    X, y = df_factors.iloc[i-back_window:i].to_numpy(), df_tags.iloc[i-back_window:i].to_numpy().reshape(-1)
    if with_ts_z_score:
        X = ts_z_score(pd.DataFrame(X)).to_numpy()
    if with_pca:
        X = fixed_pca(X)
    return X, y, [df_factors.index[i]]

def get_model_train_and_test(handler, model_id, X, y):
    X_train, X_test = X[:-1, :], X[-1, :].reshape(1, -1)
    y_train, y_test = y[:-1], y[-1] 
    handler.train_model(model_id, X_train, y_train)

    if model_id.split('_')[0] == 'StatsOLSLRModel':
        selected_indices = np.where(handler._models[model_id]._model.pvalues[1:] < 0.05)[0]
        if len(selected_indices) > 0:
            X_train = X_train[:, np.ix_(selected_indices)[0]]
            
            # 筛选完特征后再fit
            handler.train_model(model_id, X_train, y_train)
            X_test = X_test[:, np.ix_(selected_indices)[0]]
    elif model_id.split('_')[0] == 'LassoLRModel':
        pass
    return handler.evaluate_model(model_id, X_test, y_test)

def run_model(handler, model_name, model_id, *args):
    model_id, model = handler.new_model(model_name, model_id)
    X, y, t_index = get_data_split(*args)
    preds, rank_ic = get_model_train_and_test(handler, model_id, X, y)
    df_preds = pd.DataFrame(index=t_index)
    df_preds['preds'] = preds
    return df_preds

def rolling_run_models(model_name, model_id, df_factors, df_tags, back_window, with_ts_z_score, with_pca, use_parallel=True) -> pd.DataFrame:
    df_factors.fillna(0, inplace=True)
    df_tags.fillna(0, inplace=True)

    handler = Handler()
    
    if use_parallel:
        list_preds = joblib.Parallel(n_jobs=-1, backend='loky', verbose=0)( \
                        joblib.delayed(run_model)(handler, model_name, model_id, df_factors, df_tags, i, back_window, with_ts_z_score, with_pca)\
                        for i in range(back_window+1, len(df_factors.index)))
    else:
        list_preds = [run_model(handler, model_name, model_id, df_factors, df_tags, i, back_window, with_ts_z_score, with_pca)\
                      for i in range(back_window+1, len(df_factors.index))]
    df_preds = pd.concat(list_preds)
    df_preds.sort_index(inplace=True)
    return df_preds