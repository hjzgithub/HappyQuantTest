from portfolio_manager.Handler import Handler
from utils.pca_tools import fixed_pca
import pandas as pd
import joblib 

def get_data_split(df_factors, df_tags, i, back_window, with_pca):
    X, y = df_factors.iloc[i-back_window:i].to_numpy(), df_tags.iloc[i-back_window:i].to_numpy().reshape(-1)
    if with_pca:
        X = fixed_pca(X)
    return X, y, [df_factors.index[i]]

def get_model_train_and_test(model, X, y):
    X_train, X_test = X[:-1, :], X[-1, :].reshape(1, -1)
    y_train, y_test = y[:-1], y[-1] 
    model.build_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds

def run_model(model_name, *args):
    model = Handler.new_model(model_name)
    X, y, t_index = get_data_split(*args)
    preds = get_model_train_and_test(model, X, y)
    df_preds = pd.DataFrame(index=t_index)
    df_preds['preds'] = preds
    return df_preds

def rolling_run_models(model_name, df_factors, df_tags, back_window, with_pca=False, use_parallel=True) -> pd.DataFrame:
    df_factors.fillna(0, inplace=True)
    df_tags.fillna(0, inplace=True)

    if use_parallel:
        list_preds = joblib.Parallel(n_jobs=-1, backend='loky', verbose=0)( \
                        joblib.delayed(run_model)(model_name, df_factors, df_tags, i, back_window, with_pca)\
                        for i in range(back_window+1, len(df_factors.index)))
    else:
        list_preds = [run_model(model_name, df_factors, df_tags, i, back_window, with_pca)\
                      for i in range(back_window+1, len(df_factors.index))]
    df_preds = pd.concat(list_preds)
    df_preds.sort_index(inplace=True)
    return df_preds