class MyLogReg:
    """Логистическая регрессия"""

    n_iter: int = 10
    """Количество шагов градиентного спуска"""

    learning_rate: float = 0.1
    """Коэффициент скорости обучения градиентного спуска"""

    def __init__(self, **kw):
        self.n_iter = kw.get("n_iter", self.n_iter)
        self.learning_rate = kw.get("learning_rate", self.learning_rate)

    def __repr__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
