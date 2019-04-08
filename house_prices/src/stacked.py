import numpy as np


class AveragingModels():
    def __init__(self, names, models):
        self.models = models
        self.names = names

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        # Train cloned base models
        for model in self.models:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        return np.mean(predictions, axis=1)

    def score(self, test_X, test_Y):
        scores = [model.score(test_X, test_Y) for model in self.models]
        return sum(scores) / len(scores)
