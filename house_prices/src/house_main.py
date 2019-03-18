import io
import pickle
from datetime import datetime
from math import log

import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

COL_Y = 'salepricelog'
COL_ID = 'passengerid'

TEST_PATH = '../data/test.csv'
TRAIN_PATH = '../data/train.csv'
MODEL_PATH = '../model/{}.dat'
SUBMISSION_PATH = '../output/my_submission.csv'

SEED = 42


def debug(msg):
    print("{} ===>\n{}".format(datetime.now(), msg))


class House:

    def __init__(self):
        self.d = pd.read_csv(TRAIN_PATH)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        rcParams.update({'figure.autolayout': True})

    def general(self):
        debug("general")

        with open("../output/general/head.txt", "w") as file:
            file.write(str(self.d.head()))

        with open("../output/general/desc.txt", "w") as file:
            file.write(str(self.d.describe()))

        with open("../output/general/desc_objects.txt", "w") as file:
            file.write(str(self.d.describe(include='O')))

        buf = io.StringIO()
        self.d.info(buf=buf)
        with open("../output/general/info.txt", "w") as file:
            file.write(buf.getvalue())

    def prepare_data(self, d):
        x = d.copy()
        x.columns = [n.lower() for n in x.columns]

        if 'saleprice' in x:
            x[COL_Y] = x.saleprice.apply(lambda c: log(c, 10))
            x.drop('saleprice', axis=1, inplace=True)

        x = x._get_numeric_data()
        x.fillna(x.mean(), inplace=True)
        debug(x.head())
        return x

    def train_model(self, name, model, test_size=0.3):
        prepared = self.prepare_data(self.d)

        debug("splitting")
        prepared = prepared.reindex(np.random.RandomState(seed=SEED).permutation(prepared.index))
        y = np.array(prepared[COL_Y])
        prepared = prepared.drop(COL_Y, 1)

        X = np.array(prepared)
        train_X, test_X, train_y, test_y = \
            train_test_split(X, y, test_size=test_size, random_state=SEED)

        debug("fit model {}".format(name))
        model.fit(train_X, train_y)

        if test_size > 0:
            score = model.score(test_X, test_y)
            debug("model {} score: {}".format(name, score))
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame(model.feature_importances_, index=prepared.columns, columns=['importance'])
                importance = importance.sort_values('importance', ascending=False)
                debug("importance {}".format(importance))

        with open(MODEL_PATH.format(name), 'wb') as f:
            pickle.dump(model, f)

    def predict_submission(self, name):
        debug("predict start")
        with open(MODEL_PATH.format(name), 'rb') as f:
            model = pickle.load(f)
        test = pd.read_csv(TEST_PATH)
        prepared = self.prepare_data(test)

        predictions = model.predict(prepared)
        debug(predictions)
        prices = [10 ** p for p in predictions]
        submission = pd.DataFrame(data={'SalePrice': prices})
        submission.index.names = ['Id']
        submission.index += 1461
        submission.to_csv(SUBMISSION_PATH, header=True, index=True)


h = House()
# h.general()
model = RandomForestRegressor(n_estimators=100)
classifier_name = "random_forest"
h.train_model("%s" % classifier_name, model)
h.train_model(classifier_name, model, test_size=0)
h.predict_submission(classifier_name)
