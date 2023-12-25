import os
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn import linear_model
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from models.ModelBase import ModelBase

class OLSLRModel(ModelBase):
    def __init__(self, **kwargs):
        super(OLSLRModel, self).__init__(**kwargs)
        model_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(model_dir, "configs", "OLSLRModel.yaml"), 'r') as file:
            params = yaml.safe_load(file)
        self.set_params(**params)

    def build_model(self):
        params = self._params
        self._model = linear_model.LinearRegression(**params)
        return self._model

    def fit(self, X, y, profile=[]):
        self._model.fit(X, y)
        return self._model

    def predict(self, X):
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