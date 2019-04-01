import io
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import log
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

COL_Y = 'salepricelog'
COL_ID = 'passengerid'

TEST_PATH = '../data/test.csv'
TRAIN_PATH = '../data/train.csv'
MODEL_PATH = '../output/model/{}.dat'
SUBMISSION_PATH = '../output/my_submission.csv'

SEED = 42


def debug(msg):
    print("{} ===>\n{}".format(datetime.now(), msg))


class House:

    def __init__(self):
        self.d = pd.read_csv(TRAIN_PATH)
        self.dp = self.prepare_data(self.d, False)
        self.numeric_columns = self.dp._get_numeric_data().columns
        self.categoric_columns = list(set(self.dp.columns) - set(self.numeric_columns))
        self.prepared_columns = {}
        pd.set_option('display.max_rows', 2000)
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 10000)
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        rcParams.update({'figure.autolayout': True})
        # plt.rcParams['figure.figsize'] = (1000, 1000)

    def prepare_data(self, d, final_data=True, suffix='', remove_low=True):
        x = d.copy()
        x.columns = [n.lower() for n in x.columns]

        x['mssubclass'] = x['mssubclass'].astype('category')

        x['yearremodadd_real'] = x['yearremodadd'] - x['yearbuilt']
        x['yearremodadd_real'] = np.where(x['yearremodadd_real'] == 0, 0, x['yearremodadd'])
        x.drop('yearremodadd', axis=1, inplace=True)

        x['total_sf'] = x['totalbsmtsf'] + x['1stflrsf'] + x['2ndflrsf']

        if 'saleprice' in x:
            x[COL_Y] = x.saleprice.apply(lambda c: log(c, 10))
            x.drop('saleprice', axis=1, inplace=True)

        for c in ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']:
            x['condition_join_{}'.format(c)] = np.where((x['condition1'] == c) | (x['condition2'] == c), 1, 0)
        x.drop('condition1', axis=1, inplace=True)
        x.drop('condition2', axis=1, inplace=True)

        x['bsmtfintype_none'.format(c)] = np.where((x['bsmtfintype1'] == 'Unf') & (x['bsmtfintype2'] == 'Unf'), 1, 0)

        for c in ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ']:
            x['bsmtfintype_join_{}'.format(c)] = np.where((x['bsmtfintype1'] == c) | (x['bsmtfintype2'] == c), 1, 0)
        x.drop('bsmtfintype1', axis=1, inplace=True)
        x.drop('bsmtfintype2', axis=1, inplace=True)

        for c in ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd',
                  'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing']:
            x['exterior_join_{}'.format(c)] = np.where((x['exterior1st'] == c) | (x['exterior2nd'] == c), 1, 0)
        x.drop('exterior1st', axis=1, inplace=True)
        x.drop('exterior2nd', axis=1, inplace=True)

        if final_data:

            x = pd.get_dummies(x)

            non_common_columns = ['miscfeature_TenC', 'housestyle_2.5Fin', 'electrical_Mix', 'roofmatl_Metal',
                                  'roofmatl_Roll', 'heating_OthW', 'roofmatl_ClyTile', 'poolqc_Fa', 'heating_Floor',
                                  'utilities_NoSeWa', 'roofmatl_Membran', 'garagequal_Ex', 'mssubclass_150']

            for c in non_common_columns:
                if c in x:
                    x.drop(c, axis=1, inplace=True)

            x.fillna(x.mean(), inplace=True)
            self.prepared_columns[suffix] = list(x)
            if remove_low:
                with open("../output/text/importance_low_manual.txt") as file:
                    for c in file.read().splitlines():
                        if c in x:
                            x.drop(c, axis=1, inplace=True)

            with open("../output/text/prepared_data_{}.txt".format(suffix), "w") as file:
                file.write(str(x.head(2000)))
        return x

    def general(self):
        debug("general")

        with open("../output/text/head.txt", "w") as file:
            file.write(str(self.d.head(1000)))

        with open("../output/text/desc.txt", "w") as file:
            file.write(str(self.d.describe()))

        with open("../output/text/desc_objects.txt", "w") as file:
            file.write(str(self.d.describe(include='O')))

        buf = io.StringIO()
        self.d.info(buf=buf)
        with open("../output/text/info.txt", "w") as file:
            file.write(buf.getvalue())

    def train_model(self, name, model, test_size=0.3):
        prepared = self.prepare_data(self.d, suffix="train")
        prepared = self.remove_outsiders(prepared)

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
                with open("../output/text/importance_low.txt", "w") as file:
                    for i, row in importance.iterrows():
                        if row.importance < 0.001:
                            file.write(i + "\n")
                with open("../output/text/importance.txt", "w") as file:
                    file.write(str(importance))

        with open(MODEL_PATH.format(name), 'wb') as f:
            pickle.dump(model, f)

    def prepared_columns_diff(self):
        debug('columns diff')
        train_cols = self.prepared_columns["train"]
        predict_cols = self.prepared_columns["predict"]

        diff = set(train_cols).symmetric_difference(set(predict_cols))
        diff.remove(COL_Y)
        if len(diff) > 0:
            debug("mismatching columns located")
            debug(diff)

    def predict_submission(self, name):
        debug("predict start")
        with open(MODEL_PATH.format(name), 'rb') as f:
            model = pickle.load(f)
        test = pd.read_csv(TEST_PATH)
        prepared = self.prepare_data(test, suffix="predict")

        self.prepared_columns_diff()

        predictions = model.predict(prepared)
        debug(predictions)
        prices = [10 ** p for p in predictions]
        submission = pd.DataFrame(data={'SalePrice': prices})
        submission.index.names = ['Id']
        submission.index += 1461
        submission.to_csv(SUBMISSION_PATH, header=True, index=True)

    def catplot_per_category(self, df, category, prefix="", bins=30):
        debug("catplot for {}".format(category))
        sns.catplot(x=COL_Y, y=category, kind="violin", inner="stick", data=df, height=5).fig.savefig(
            "../output/graph/{}catplot_{}.png".format(prefix, category)
        )

    def pairplot_for_numeric_column(self, df, c):
        debug("pairplot for {}".format(c))
        sns.pairplot(df[[c, COL_Y]], markers='+', height=5, diag_kws=dict(bins=30)).fig.savefig(
            "../output/graph/pair_plot_{}.png".format(c))

    def catplot_for_categories(self):
        for c in self.categoric_columns:
            self.catplot_per_category(self.dp, c)

    def pairplot_for_numeric(self):
        for c in self.numeric_columns:
            if c != COL_Y:
                self.pairplot_for_numeric_column(self.dp, c)

    def model_hyperspace(self, model, hyperspace_params):
        df = self.prepare_data(self.d, suffix="predict")

        debug("splitting")
        df = df.reindex(np.random.RandomState(seed=SEED).permutation(df.index))
        y = np.array(df[COL_Y])
        df = df.drop(COL_Y, 1)

        X = np.array(df)
        search = RandomizedSearchCV(estimator=model, param_distributions=hyperspace_params, cv=4, verbose=10,
                                    random_state=SEED, n_iter=100, n_jobs=-1)
        search.fit(X, y)
        debug(search.best_params_)

    def random_forest_hyperspace(self):
        hyperspace_params = {
            'bootstrap': [True, False],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [100, 200, 400, 600, 800]
        }
        self.model_hyperspace(RandomForestRegressor(random_state=SEED), hyperspace_params)
        # {'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 100,
        #  'bootstrap': False}

    def remove_outsiders(self, prepared):
        x = prepared
        out = x[(x['grlivarea'] < 4000) | (x[COL_Y] > 5.5)]
        return out

    def missing_values(self):
        x = self.d
        all_data_na = (x.isnull().sum() / len(x)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        with open("../output/text/missing_values.txt", "w") as file:
            file.write(str(missing_data))

        fig, ax = plt.subplots(figsize=(15, 12))
        plt.xticks(rotation='90')
        sns.barplot(x=all_data_na.index, y=all_data_na)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        fig.savefig("../output/graph/missing_values.png")


h = House()
# h.general()
# h.missing_values()
# h.catplot_for_categories()
# h.pairplot_for_numeric()
# h.random_forest_hyperspace()
# h.find_outsiders()
classifier_name = "random_forest"
model = RandomForestRegressor(n_estimators=800,
                              min_samples_split=2,
                              min_samples_leaf=1,
                              max_features='sqrt',
                              max_depth=100,
                              bootstrap=False,
                              random_state=SEED)
h.train_model("%s" % classifier_name, model)
h.train_model(classifier_name, model, test_size=0)
h.predict_submission(classifier_name)
