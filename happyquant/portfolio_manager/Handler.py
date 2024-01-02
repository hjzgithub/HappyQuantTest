from models.OLSLRModel import OLSLRModel
from models.LogisticModel import LogisticModel

class Handler:
    @staticmethod
    def new_model(model_name):
        if model_name == 'OLSLRModel':
            model = OLSLRModel()
        elif model_name == 'LogisticModel':
            model = LogisticModel()
        return model