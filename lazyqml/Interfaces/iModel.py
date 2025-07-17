from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

    @property
    @abstractmethod
    def n_params(self):
        pass