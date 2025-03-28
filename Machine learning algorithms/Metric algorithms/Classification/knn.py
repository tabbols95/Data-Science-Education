# external
import numpy as np
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

    def predict(self, features: pd.DataFrame):
        predictions = []
        for _, test_row in features.iterrows():
            distances = []
            for _, train_row in self.X.iterrows():
                distance = np.sqrt(np.sum((test_row - train_row) ** 2))
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
                distance = np.sqrt(np.sum((test_row - train_row) ** 2))
                distances.append(distance)

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y.iloc[nearest_indices]

            proba = np.mean(nearest_labels == 1)
            probas.append(proba)

        return np.array(probas)
