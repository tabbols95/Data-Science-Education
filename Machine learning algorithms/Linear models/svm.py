# external
from random import sample

import pandas as pd
import numpy as np
from typing import Union
import random


class SVM:
    """Support Vector Machine"""

    def __init__(self,
                 n_iter: int = 10,
                 learning_rate: float = 0.001,
                 C: float = 1,
                 sgd_sample: Union[int, float, None] = None,
                 random_state: int = 42,
                 **kw):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.b = None
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        params = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f'{self.__class__.__name__} class: {params}'

    def __str__(self):
        return self.__repr__()

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False):
        random.seed(self.random_state)

        observation_count, feature_count = X.shape
        self.b = 1
        self.weights = np.ones(shape=feature_count)
        target_converted = np.where(y == 0, -1, 1)

        loss = self._calculate_loss(X, target_converted)
        if verbose:
            print(f"start | loss: {loss:.2f}")

        # Формируем кол-во наблюдений в выборке
        if isinstance(self.sgd_sample, int):
            n_observations = self.sgd_sample
        elif isinstance(self.sgd_sample, float):
            n_observations = round(observation_count * self.sgd_sample)
        else:
            n_observations = observation_count

        # Выполняем n_iter шагов градиентного спуска
        for i in range(self.n_iter):
            if self.sgd_sample is None:
                sample_rows_idx = range(n_observations)
            else:
                sample_rows_idx = random.sample(range(observation_count), n_observations)
            X_sample = X.iloc[sample_rows_idx].reset_index(drop=True)
            target_converted_sample = target_converted[sample_rows_idx]

            for i_row in range(n_observations):
                x_i = X_sample.iloc[i_row].values
                y_i = target_converted_sample[i_row]

                if y_i * (x_i @ self.weights + self.b) >= 1:
                    grad_w = 2 * self.weights
                    grad_b = 0
                else:
                    grad_w = 2 * self.weights - self.C * y_i * x_i
                    grad_b = - self.C * y_i

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
        hinge_loss = self.C * np.maximum(0, 1 - y * (X @ self.weights + self.b)).mean()
        regularization = np.linalg.norm(self.weights) ** 2
        return regularization + hinge_loss

    def get_coef(self) -> tuple:
        return self.weights, self.b
