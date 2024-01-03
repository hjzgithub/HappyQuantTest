import os
import joblib
import yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ModelBase import ModelBase

class StatsOLSLRModel(ModelBase):
    def __init__(self, **kwargs):
        super(StatsOLSLRModel, self).__init__(**kwargs)
        model_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(model_dir, "configs", "StatsOLSLRModel.yaml"), 'r') as file:
            params = yaml.safe_load(file)
        self.set_params(**params)

    def build_model(self):
        params = self._params
        self.fit_intercept = params['fit_intercept']
        self._model = None
        return self._model

    def fit(self, X, y, profile=[]):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self._model = sm.OLS(y, X).fit()
        return self._model

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        preds = self._model.predict(X)
        return preds
    
    def load_model(self, model_path: str):
        self._model = joblib.load(model_path)
        return self._model
    
    def save_model(self, model_path: str):
        joblib.dump(self._model, model_path)
        return self._model
    
    @staticmethod
    def model_type() -> str:
        return "Regression"

    @staticmethod
    def data_type() -> str:
        return "numpy"
    
    @staticmethod
    def eval_metric() -> str:
        return "MSE"
    
    @staticmethod
    def n_features() -> int:
        return 306