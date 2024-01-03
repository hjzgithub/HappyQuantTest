import importlib
import scipy

class Handler:
    def __init__(self):
        self._models = dict()

    def new_model(self, model_name, model_id, params=None):
        # Generate model instance
        model_file = ".".join(["models", model_name])
        model_module = importlib.import_module(model_file)
        model_class = getattr(model_module, model_name)
        self._models[model_id] = model_class()
        if params:
            self._models[model_id].set_params(**params)
        self._models[model_id].build_model()
        return model_id, self._models[model_id]
    
    def train_model(self, model_id, X, y):
        assert model_id in self._models.keys()
        self._models[model_id].fit(X, y)
        return self._models[model_id]
    
    def evaluate_model(self, model_id, X, y):
        assert model_id in self._models.keys()
        preds = self._models[model_id].predict(X)
        return preds, scipy.stats.spearmanr(preds, y)[0]