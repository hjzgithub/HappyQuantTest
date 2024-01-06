import os
import abc
import yaml
from typing import Any, Dict

class StrategyTemplate(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        strategy_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(strategy_dir, "configs", "StrategyTemplate.yaml"), 'r') as f:
            baseArgs = yaml.safe_load(f)
        self.__dict__.update(baseArgs)

    def set_params(self, **params: Any) -> Dict[str, Any]:
        if not hasattr(self, "_params"):
            self._params = dict()
        self._params.update(params)
        return self._params