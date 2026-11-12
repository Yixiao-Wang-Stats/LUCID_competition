#  This file is an abstract base class for all predictive models 
from abc import ABC, abstractmethod


class PredictiveModel(ABC):
    def __init__(self):
        self.is_fitted = False


    def __call__(
        self,
        X,
    ):
        return self.predict(X)

    @abstractmethod
    def fit(
        self,
        X_train,
        y_train,
    ):
        pass


    @abstractmethod
    def predict(
        self,
        X,
    ):
        pass


    @abstractmethod
    def save(
        self,
        path,
    ):
        pass


    @classmethod
    @abstractmethod
    def load(
        cls,
        path,
    ):
        pass