import pandas as pd
import numpy as np
from typing import Optional


class MyLineReg:
    """Линейная регрессия"""

    n_iter: int
    """Количество шагов градиентного спуска"""

    learning_rate: float
    """Коэффициент скорости обучения градиентного спуска"""

    weights: Optional[list]
    """Веса модели"""

    MSE: Optional[float]
    """Среднеквадратичная ошибка"""

    def __init__(self, **kwargs):
        self.n_iter = kwargs.get("n_iter")
        self.learning_rate = kwargs.get("learning_rate")
        self.weights = None
        self.MSE = None

    def __repr__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series):

        observation_count, feature_count = X.shape
        X.insert(0, "x_0", 1)  # свободный коэффициент
        self.weights = np.ones(feature_count + 1)

        for i in range(self.n_iter):
            y_predict = (X * self.weights).sum(axis=1)
            self.MSE = ((y_predict - y) ** 2).sum() / observation_count
            gradient = 2 / observation_count * np.dot(X.T, (y_predict - y))
            self.weights -= self.learning_rate * gradient

    def predict(self, X: pd.DataFrame):
        X.insert(0, "x_0", 1)
        y_predict = (X * self.weights).sum(axis=1)
        return y_predict

    def get_coef(self):
        return self.weights[1:]
