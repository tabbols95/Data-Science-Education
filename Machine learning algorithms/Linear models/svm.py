class SVM:
    """Support Vector Machine"""

    def __init__(self, n_iter: int = 10, learning_rate: float = 0.001, **kw):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __repr__(self):
        params = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f'{self.__class__.__name__} class: {params}'

    def __str__(self):
        return self.__repr__()

if __name__ == '__main__':
    my_svm = SVM()
    print(my_svm)
