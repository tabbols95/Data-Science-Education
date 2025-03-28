# external
import numpy as np
import pandas as pd
from typing import Literal


class KNNClassification:
    """Метод ближайших соседей (Классификация)"""

    def __init__(self, k: int = 3,
                 metric: Literal['euclidean', 'chebyshev', 'manhattan', 'cosine'] = 'euclidean'):
        self.k = k
        self.train_size = None
        self.X = pd.DataFrame()
        self.y = pd.Series()
        self.metric = metric

    def __repr__(self):
        params = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f'{self.__class__.__name__} class: {params}'

    def __str__(self):
        return self.__repr__()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape

    def predict(self, features: pd.DataFrame):
        predictions = []
        for _, test_row in features.iterrows():
            distances = []
            for _, train_row in self.X.iterrows():
                distance = self._calc_distance(test_row, train_row)
                distances.append(distance)

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y.iloc[nearest_indices]

            # Определяем модальный класс
            counts = nearest_labels.value_counts()
            if len(counts) == 2 and counts[0] == counts[1]:
                pred = 1
            else:
                pred = counts.idxmax()

            predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, features: pd.DataFrame):
        probas = []
        for _, test_row in features.iterrows():
            distances = []
            for _, train_row in self.X.iterrows():
                distance = self._calc_distance(test_row, train_row)
                distances.append(distance)

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y.iloc[nearest_indices]

            proba = np.mean(nearest_labels == 1)
            probas.append(proba)

        return np.array(probas)

    def _calc_distance(self,
                       test_row: pd.Series,
                       train_row: pd.Series) -> float:

        if self.metric == 'euclidean':
            return np.sqrt(np.sum((test_row - train_row) ** 2))
        elif self.metric == 'chebyshev':
            return np.linalg.norm(test_row - train_row, ord=np.inf)
        elif self.metric == 'manhattan':
            return np.linalg.norm(test_row - train_row, ord=1)
        elif self.metric == 'cosine':
            return 1 - (np.sum(test_row * train_row) / (np.sqrt(np.sum(test_row ** 2)) * np.sqrt(np.sum(train_row ** 2))))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
