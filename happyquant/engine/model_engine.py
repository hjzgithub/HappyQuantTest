from model_manager.Handler import Handler
from model_manager.feature_engineer import *
from utils.cal_tools import *
import pandas as pd
import joblib 

class ModelEngine:
    def __init__(self):
        self.handler = Handler()

    def get_model_train_and_test(self, model_id, X, y):
        X_train, X_test = X[:-1, :], X[-1, :].reshape(1, -1)
        y_train, y_test = y[:-1], y[-1] 
        self.handler.train_model(model_id, X_train, y_train)
        return self.handler.evaluate_model(model_id, X_test, y_test)

    def run_model(self, model_name, model_id, *args):
        model_id, model = self.handler.new_model(model_name, model_id)
        X, y, t_index = get_data_split(*args)
        preds, rank_ic = self.get_model_train_and_test(model_id, X, y)
        df_preds = pd.DataFrame(index=t_index)
        df_preds['preds'] = preds
        return df_preds

    def rolling_run_models(self, 
                        model_name, 
                        model_id, 
                        df_factors, 
                        df_tags, 
                        back_window, 
                        target_type, 
                        standardize_method, 
                        single_test_method, 
                        combine_method, 
                        multi_test_method,
                        use_parallel=True,
                        ) -> pd.DataFrame:
        df_factors.fillna(0, inplace=True)
        df_tags.fillna(0, inplace=True)

        # debug
        #use_parallel = False
        if use_parallel:
            list_preds = joblib.Parallel(n_jobs=-1, backend='loky', verbose=0)( \
                            joblib.delayed(self.run_model)(model_name, model_id, df_factors, df_tags, i, \
                                                    back_window, target_type, standardize_method, single_test_method, combine_method, multi_test_method)\
                            for i in range(back_window+1, len(df_factors.index)))
        else:
            list_preds = [self.run_model(model_name, model_id, df_factors, df_tags, i, \
                                    back_window, target_type, standardize_method, single_test_method, combine_method, multi_test_method)\
                        for i in range(back_window+1, len(df_factors.index))]
        df_preds = pd.concat(list_preds)
        df_preds.sort_index(inplace=True)
        return df_preds
    
def get_data_split(df_factors, df_tags, i, back_window, target_type, standardize_method, single_test_method, combine_method, multi_test_method):
    X, y = df_factors.iloc[i-back_window:i].to_numpy(), df_tags.iloc[i-back_window:i].to_numpy().reshape(-1)
    y = trans_tag(y, target_type)
    X = standardize_feature(X, standardize_method)
    X = single_factor_test(X, y, single_test_method)
    X = combine_feature(X, y, combine_method)
    X = multi_factor_test(X, y, multi_test_method)
    return X, y, [df_factors.index[i]]