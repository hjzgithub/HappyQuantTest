import os
import abc
import yaml
from typing import Any, Dict

class ModelBase(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        model_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(model_dir, "configs", "ModelBase.yaml"), 'r') as f:
            baseArgs = yaml.safe_load(f)
        self.__dict__.update(baseArgs)

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def save_model(self):
        pass

    @abc.abstractmethod
    def model_type(self):
        pass

    @abc.abstractmethod
    def eval_metric(self):
        pass

    @abc.abstractmethod
    def data_type(self):
        pass

    @abc.abstractmethod
    def n_features(self):
        pass

    def get_params(self) -> Dict[str, Any]:
        return self._params

    def set_params(self, **params: Any) -> Dict[str, Any]:
        if not hasattr(self, "_params"):
            self._params = dict()
        self._params.update(params)
        return self._params