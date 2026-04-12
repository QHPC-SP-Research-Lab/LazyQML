from abc import ABC, abstractmethod

class Preprocessing(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        pass
    
    @abstractmethod
    def fit_transform(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass