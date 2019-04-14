import numpy as np


class StackedAverage():
    def __init__(self,  models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        return np.mean(predictions, axis=1)

    def score(self, test_X, test_Y):
        scores = [model.score(test_X, test_Y) for model in self.models]
        return sum(scores) / len(scores)
