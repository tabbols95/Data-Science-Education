import random
import pandas as pd
import numpy as np
from typing import Optional, Literal, Union


class MyLineReg:
    """Линейная регрессия"""

    n_iter: int = 1000  # Значение по умолчанию
    """Количество шагов градиентного спуска"""

    learning_rate: float = 0.01  # Значение по умолчанию
    """Коэффициент скорости обучения градиентного спуска"""

    metric: Optional[Literal["mae", "mse", "rmse", "mape", "r2"]] = "mse"
    """Метрика оценки точности модели"""

    weights: Optional[np.ndarray] = None
    """Веса модели"""

    sgd_sample: Union[int, float, None] = None
    """Кол-во образцов, которое будет использовано на каждой итерации обучения"""

    random_state: int = 42
    """Параметр случайности"""

    def __init__(self, **kwargs):
        self.n_iter = kwargs.get("n_iter", self.n_iter)
        self.learning_rate = kwargs.get("learning_rate", self.learning_rate)
        self.metric = kwargs.get("metric", self.metric)
        self._metric_value = None
        self._loss_value = None
        self.weights = None
        self.sgd_sample = kwargs.get("sgd_sample", self.sgd_sample)
        self.random_state = kwargs.get("random_state", self.random_state)

        # регуляризация
        self.reg = kwargs.get("reg")
        self.l1_coef = 0
        self.l2_coef = 0

        if self.reg == "l1" or self.reg == "elasticnet":
            self.l1_coef = kwargs.get("l1_coef")

        if self.reg == "l2" or self.reg == "elasticnet":
            self.l2_coef = kwargs.get("l2_coef")

    def __repr__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):
        observation_count, feature_count = X.shape
        X.insert(0, "x_0", 1)  # свободный коэффициент
        self.weights = np.ones(feature_count + 1)

        if isinstance(self.sgd_sample, int):
            sgd_sample = self.sgd_sample
        elif isinstance(self.sgd_sample, float):
            sgd_sample = int(self.sgd_sample * observation_count)
        else:
            sgd_sample = observation_count

        random.seed(self.random_state)

        for i in range(self.n_iter):
            sample_rows_idx = random.sample(range(observation_count), sgd_sample)
            X_sample = X.iloc[sample_rows_idx]
            y_sample = y.iloc[sample_rows_idx]
            observation_count_sample, feature_count_sample = X_sample.shape

            y_predict = (X_sample.values @ self.weights)  # Используем матричное умножение
            self._loss_value = ((y_sample - y_predict) ** 2).sum() / observation_count_sample + self.l1_coef * self.weights.sum() + self.l2_coef * (self.weights ** 2).sum()  # mse

            gradient = 2 / observation_count_sample * X_sample.T.values @ (y_predict - y_sample) + self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
            self.weights -= (self.learning_rate(i+1) if callable(self.learning_rate) else self.learning_rate) * gradient

        y_predict = (X.values @ self.weights)

        self._metric_value = {
            "mae": (y - y_predict).abs().sum() / observation_count,
            "mse": ((y - y_predict) ** 2).sum() / observation_count,
            "rmse": np.sqrt(((y - y_predict) ** 2).sum() / observation_count),
            "mape": 100 / observation_count * ((y - y_predict) / y).abs().sum(),
            "r2": 1 - (((y - y_predict) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        }[self.metric]

    def predict(self, X: pd.DataFrame):
        if "x_0" not in X.columns:
            X.insert(0, "x_0", 1)
        y_predict = (X.values @ self.weights).sum()
        return y_predict

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self._metric_value