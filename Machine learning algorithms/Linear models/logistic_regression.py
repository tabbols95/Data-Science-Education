from typing import Optional, Literal
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class MyLogReg:
    """Логистическая регрессия"""

    n_iter: int = 10
    """Количество шагов градиентного спуска"""

    learning_rate: float = 0.1
    """Коэффициент скорости обучения градиентного спуска"""

    weights: Optional[np.ndarray] = None
    """Веса модели"""

    metrics: dict = {}
    """Метрики обученной модели"""

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

        y_predict_bin = [int(y_pred) for y_pred in (1 / (1 + np.exp(-(X @ self.weights))) > 0.5)]
        self.metrics["accuracy"] = accuracy_score(y, y_predict_bin)
        self.metrics["precision"] = precision_score(y, y_predict_bin)
        self.metrics["recall"] = recall_score(y, y_predict_bin)
        self.metrics["f1"] = f1_score(y, y_predict_bin)
        self.metrics["roc_auc"] = roc_auc_score(y, y_predict_bin)

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

    def get_best_score(self, metric: Optional[Literal["accuracy", "precision", "recall", "f1", "roc_auc"]] = None):
        return self.metrics.get(metric)

    def get_coef(self):
        return self.weights[1:]