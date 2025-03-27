# external
import pandas as pd
import numpy as np
from typing import Union


class SVM:
    """Support Vector Machine"""

    def __init__(self,
                 n_iter: int = 10,
                 learning_rate: float = 0.001,
                 **kw):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.b = None

    def __repr__(self):
        params = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f'{self.__class__.__name__} class: {params}'

    def __str__(self):
        return self.__repr__()

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False):
        observation_count, feature_count = X.shape
        self.b = 1
        self.weights = np.ones(shape=feature_count)
        target_converted = np.where(y == 0, -1, 1)

        loss = self._calculate_loss(X, target_converted)
        if verbose:
            print(f"start | loss: {loss:.2f}")

        # Выполняем n_iter шагов градиентного спуска
        for i in range(self.n_iter):
            for i_row in range(observation_count):
                x_i = X.iloc[i_row].values
                y_i = target_converted[i_row]

                if y_i * (x_i @ self.weights + self.b) >= 1:
                    grad_w = 2 * self.weights
                    grad_b = 0
                else:
                    grad_w = 2 * self.weights - y_i * x_i
                    grad_b = - y_i

                self.weights -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b

            # Вывод логов
            if verbose and (i % verbose == 0 or i == self.n_iter):
                loss = self._calculate_loss(X, target_converted)
                print(f"{i} | loss: {loss:.2f}")

    def predict(self, features: pd.DataFrame):
        y = np.sign(features @ self.weights + self.b)
        return np.where(y == -1, 0, 1)

    def _calculate_loss(self, X: pd.DataFrame, y: np.array) -> float:
        """Вычисление функции потерь"""
        hinge_loss = np.maximum(0, 1 - y * (X @ self.weights + self.b)).mean()
        regularization = np.linalg.norm(self.weights) ** 2
        return regularization + hinge_loss

    def get_coef(self) -> tuple:
        return self.weights, self.b
