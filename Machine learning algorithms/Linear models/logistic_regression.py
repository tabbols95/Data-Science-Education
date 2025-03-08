from typing import Optional
import numpy as np
import pandas as pd


class MyLogReg:
    """Логистическая регрессия"""

    n_iter: int = 10
    """Количество шагов градиентного спуска"""

    learning_rate: float = 0.1
    """Коэффициент скорости обучения градиентного спуска"""

    weights: Optional[np.ndarray] = None
    """Веса модели"""

    eps = 1e-15
    """Во избежании получения inf"""

    def __init__(self, **kw):
        self.n_iter = kw.get("n_iter", self.n_iter)
        self.learning_rate = kw.get("learning_rate", self.learning_rate)
        self._log_loss_value = None

    def __repr__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):
        observation_count, feature_count = X.shape
        X.insert(0, "x_0", 1)  # свободный коэффициент
        self.weights = np.ones(feature_count + 1)

        for i in range(self.n_iter):
            y_predict = 1 / (1 + np.exp(-(X @ self.weights)))
            self._log_loss_value = (y * np.log(y_predict + self.eps) + (1 - y) * np.log(1 - y_predict + self.eps)).sum() / observation_count
            gradient = X.T.values @ (y_predict - y) / observation_count
            self.weights -= self.learning_rate * gradient

    def predict_proba(self, X: pd.DataFrame):
        if "x_0" not in X.columns:
            X.insert(0, "x_0", 1)
        y_predict = (1 / (1 + np.exp(-(X @ self.weights)))).mean()
        return y_predict

    def predict(self, X: pd.DataFrame):
        if "x_0" not in X.columns:
            X.insert(0, "x_0", 1)
        y_predict = (1 / (1 + np.exp(-(X @ self.weights))) > 0.5).sum()
        return y_predict

    def get_coef(self):
        return self.weights[1:]