from models.OLSLRModel import OLSLRModel
from models.pca import fixed_pca
import pandas as pd

def rolling_run_models(df_factors, df_tags, back_window, with_pca=True):
    df_factors.fillna(0, inplace=True)
    df_tags.fillna(0, inplace=True)

    list_preds = []
    for i in range(back_window+1, len(df_factors.index)):
        # Data
        X, y = df_factors.iloc[i-back_window:i].to_numpy(), df_tags.iloc[i-back_window:i].to_numpy().reshape(-1)
        if with_pca:
            X = fixed_pca(X)
        X_train, X_test = X[:-1, :], X[-1, :].reshape(1, -1)
        y_train, y_test = y[:-1], y[-1] 

        mymodel = OLSLRModel()
        mymodel.build_model()
        mymodel.fit(X_train, y_train)
        preds = mymodel.predict(X_test)
        list_preds.append(preds)
    df_preds = pd.DataFrame(list_preds, index=df_factors.index[back_window+1:], columns=['preds'])
    return df_preds