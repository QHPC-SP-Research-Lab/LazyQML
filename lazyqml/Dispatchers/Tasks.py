from dataclasses import dataclass
from typing import Union

from lazyqml.Global import Model, Embedding, Ansatzs

@dataclass
class QMLTask:
    id: int
    model_memory: float
    
    X_train: any
    X_test: any
    y_train: any
    y_test: any

    custom_metric: any
    model_params: dict = None

    def get_model_params(self):
        pass

    def get_data(self):
        return (self.X_train, self.y_train, self.X_test, self.y_test)
    
    def get_task_params(self):
        # print((self.model_params, *self.get_data(), self.custom_metric))
        return (self.id, self.model_params, *self.get_data(), self.custom_metric)