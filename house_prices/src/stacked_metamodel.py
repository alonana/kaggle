import numpy as np


class StackedMetadata():
    def __init__(self,  models, meta_model):
        self.models = models
        self.meta_model = meta_model

    def create_metadata_x(self, X):
        metadata_x = np.copy(X)
        for model in self.models:
            predictions = model.predict(X)
            metadata_x = np.column_stack((metadata_x, predictions))

        return metadata_x

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

        metadata_x = self.create_metadata_x(X)
        self.meta_model.fit(metadata_x, y)

    def predict(self, X):
        metadata_x = self.create_metadata_x(X)
        return self.meta_model.predict(metadata_x)

    def score(self, test_X, test_Y):
        metadata_x = self.create_metadata_x(test_X)
        return self.meta_model.score(metadata_x, test_Y)
