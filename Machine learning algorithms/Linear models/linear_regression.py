import pandas as pd
import numpy as np
from typing import Optional, Literal


class MyLineReg:
    """Линейная регрессия"""

    n_iter: int = 1000  # Значение по умолчанию
    """Количество шагов градиентного спуска"""

    learning_rate: float = 0.01  # Значение по умолчанию
    """Коэффициент скорости обучения градиентного спуска"""

    metric: Optional[Literal["mae", "mse", "rmse", "mape", "r2"]] = None
    """Метрика оценки точности модели"""

    weights: Optional[np.ndarray] = None
    """Веса модели"""

    def __init__(self, **kwargs):
        self.n_iter = kwargs.get("n_iter", self.n_iter)
        self.learning_rate = kwargs.get("learning_rate", self.learning_rate)
        self.metric = kwargs.get("metric", self.metric)
        self._metric_value = None
        self._loss_value = None
        self.weights = None

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

        for i in range(self.n_iter):
            y_predict = (X.values @ self.weights)  # Используем матричное умножение
            self._loss_value = ((y - y_predict) ** 2).sum() / observation_count + self.l1_coef * self.weights.sum() + self.l2_coef * (self.weights ** 2).sum()  # mse

            gradient = 2 / observation_count * X.T.values @ (y_predict - y) + self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
            self.weights -= (self.learning_rate(i+1) if callable(self.learning_rate) else self.learning_rate) * gradient

            if verbose and (i == 0 or i % (self.n_iter // 10) == 0):  # Вывод каждые 10%
                print(f"{i} | loss: {self._loss_value:.2f}",
                      f"| {self.metric}: {self._metric_value:.2f}" if self.metric is not None else "")

        y_predict = (X.values @ self.weights)

        if self.metric == "mae":
            # mean absolute error
            self._metric_value = (y - y_predict).abs().sum() / observation_count
        elif self.metric == "rmse":
            # root mean squared error
            self._metric_value = np.sqrt(((y - y_predict) ** 2).sum() / observation_count)
        elif self.metric == "mape":
            # mean absolute percentage error
            self._metric_value = 100 / observation_count * ((y - y_predict) / y).abs().sum()
        elif self.metric == "r2":
            # coefficient of determination
            self._metric_value = 1 - (((y - y_predict) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        else:
            # mean squared error
            self._metric_value = ((y - y_predict) ** 2).sum() / observation_count

    def predict(self, X: pd.DataFrame):
        if "x_0" not in X.columns:
            X.insert(0, "x_0", 1)
        y_predict = (X.values @ self.weights)
        return y_predict

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self._metric_value