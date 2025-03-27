# external
import pandas as pd


class KNNClassification:
    """Метод ближайших соседей (Классификация)"""

    def __init__(self, k: int = 3):
        self.k = k
        self.train_size = None
        self.X = pd.DataFrame()
        self.y = pd.Series()

    def __repr__(self):
        params = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f'{self.__class__.__name__} class: {params}'

    def __str__(self):
        return self.__repr__()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape
