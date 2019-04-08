import io
import pathlib
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import log
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from house_prices.src.stacked import AveragingModels

COL_Y = 'salepricelog'

OUTPUT_PATH = "../output/"
TEST_PATH = '../data/test.csv'
TRAIN_PATH = '../data/train.csv'
IMPORTANCE_THRESHOLD = 0.001
IMPORTANCE_LOW_CURRENT = OUTPUT_PATH + "text/importance_low_current.txt"
IMPORTANCE_LOW_TOTAL = OUTPUT_PATH + "text/importance_low_total.txt"
COLUMNS_AFTER_PREPARE = OUTPUT_PATH + "text/columns_after_prepare_{}.txt"

MODEL_PATH = OUTPUT_PATH + 'model/{}.dat'
SUBMISSION_PATH = OUTPUT_PATH + 'my_submission.csv'

SEED = 42


def debug(msg):
    print("{} ===>\n{}".format(datetime.now(), msg))


class House:

    def __init__(self, data_path=TRAIN_PATH):
        pathlib.Path(OUTPUT_PATH + "graph").mkdir(parents=True, exist_ok=True)
        pathlib.Path(OUTPUT_PATH + "model").mkdir(parents=True, exist_ok=True)
        pathlib.Path(OUTPUT_PATH + "text").mkdir(parents=True, exist_ok=True)
        csv = pd.read_csv(data_path)
        self.discrete_numbers = ["bedroomabvgr", "bsmtfullbath", "bsmthalfbath", "fireplaces", "fullbath", "garagecars",
                                 "halfbath", "kitchenabvgr", "mosold", "overallcond", "overallqual", "total_bathrooms",
                                 "totrmsabvgrd", "yrsold"]

        self.continuous_numbers = ["1stflrsf", "2ndflrsf", "3ssnporch", "bsmtfinsf1", "bsmtfinsf2", "bsmtunfsf",
                                   "enclosedporch", "garagearea", "garageyrblt", "grlivarea", "lotarea", "lotfrontage",
                                   "lowqualfinsf", "masvnrarea", "miscval", "openporchsf", "poolarea", "screenporch",
                                   "total_porch_sf", "total_sf", "total_sqr_footage", "totalbsmtsf", "wooddecksf",
                                   "yearbuilt"]

        self.df = self.preprocess(csv, remove_outside=False)
        self.numeric_columns = self.df._get_numeric_data().columns
        self.categoric_columns = list(set(self.df.columns) - set(self.numeric_columns))

    def create_normalized_columns(self, df, column):
        # zscore
        df["{}_zscore".format(column)] = (df[column] - df[column].mean()) / df[column].std(ddof=0)

        # min-max
        # df["{}_minmax".format(column)] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    def join_splitted_columns(self, x):
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
        return x

    def combine_columns(self, x):
        x['total_sf'] = x['totalbsmtsf'] + x['1stflrsf'] + x['2ndflrsf']
        x['total_sqr_footage'] = (x['bsmtfinsf1'] + x['bsmtfinsf2'] +
                                  x['1stflrsf'] + x['2ndflrsf'])
        x['total_bathrooms'] = (x['fullbath'] + (0.5 * x['halfbath']) +
                                x['bsmtfullbath'] + (0.5 * x['bsmthalfbath']))
        x['total_porch_sf'] = (x['openporchsf'] + x['3ssnporch'] +
                               x['enclosedporch'] + x['screenporch'] +
                               x['wooddecksf'])
        x['haspool'] = x['poolarea'].apply(lambda c: 1 if c > 0 else 0)
        x['has2ndfloor'] = x['2ndflrsf'].apply(lambda c: 1 if c > 0 else 0)
        x['hasgarage'] = x['garagearea'].apply(lambda c: 1 if c > 0 else 0)
        x['hasbsmt'] = x['totalbsmtsf'].apply(lambda c: 1 if c > 0 else 0)
        x['hasfireplace'] = x['fireplaces'].apply(lambda c: 1 if c > 0 else 0)

    def preprocess(self, csv, remove_outside=False):
        x = csv.copy()
        x.columns = [n.lower() for n in x.columns]
        x.drop('id', axis=1, inplace=True)

        x['mssubclass'] = x['mssubclass'].astype('category')

        x['yearremodadd_real'] = x['yearremodadd'] - x['yearbuilt']
        x['yearremodadd_real'] = np.where(x['yearremodadd_real'] == 0, 0, x['yearremodadd'])
        x.drop('yearremodadd', axis=1, inplace=True)

        self.combine_columns(x)

        if 'saleprice' in x:
            x[COL_Y] = x.saleprice.apply(lambda v: log(v, 10))
            x.drop('saleprice', axis=1, inplace=True)

        if remove_outside:
            x = self.remove_outsiders(x)

        x = self.join_splitted_columns(x)

        for c in self.continuous_numbers:
            self.create_normalized_columns(x, c)
        return x

    def prepare_data(self, suffix=''):
        debug("prepare data")
        x = self.df.copy()
        for c in self.continuous_numbers:
            x.drop(c, axis=1, inplace=True)

        x = pd.get_dummies(x)

        non_common_columns = ['garagequal_Ex', 'electrical_Mix', 'roofmatl_Metal', 'miscfeature_TenC',
                              'utilities_NoSeWa', 'heating_Floor', 'roofmatl_Membran', 'poolqc_Fa', 'heating_OthW',
                              'roofmatl_ClyTile', 'roofmatl_Roll', 'housestyle_2.5Fin', 'mssubclass_150']

        for c in non_common_columns:
            if c in x:
                x.drop(c, axis=1, inplace=True)

        x.fillna(x.mean(), inplace=True)

        with open(IMPORTANCE_LOW_TOTAL) as file:
            for c in file.read().splitlines():
                if c in x:
                    x.drop(c, axis=1, inplace=True)

        with open(OUTPUT_PATH + "text/prepared_data_{}.txt".format(suffix), "w") as file:
            file.write(str(x.head(2000)))

        with open(COLUMNS_AFTER_PREPARE.format(suffix), "wb") as f:
            final_cols = list(x)
            pickle.dump(final_cols, f)

        return x

    def general(self):
        debug("general")

        with open(OUTPUT_PATH + "text/head.txt", "w") as file:
            file.write(str(self.df.head(1000)))

        with open(OUTPUT_PATH + "text/desc.txt", "w") as file:
            file.write(str(self.df.describe()))

        with open(OUTPUT_PATH + "text/desc_objects.txt", "w") as file:
            file.write(str(self.df.describe(include='O')))

        buf = io.StringIO()
        self.df.info(buf=buf)
        with open(OUTPUT_PATH + "text/info.txt", "w") as file:
            file.write(buf.getvalue())

    def train_model(self, name, model, test_size=0.3):
        prepared = self.prepare_data(suffix="train")

        prepared = prepared.reindex(np.random.RandomState(seed=SEED).permutation(prepared.index))
        y = np.array(prepared[COL_Y])
        prepared = prepared.drop(COL_Y, 1)

        X = np.array(prepared)
        train_X, test_X, train_y, test_y = \
            train_test_split(X, y, test_size=test_size, random_state=SEED)

        if name is not None:
            debug("fit model {}".format(name))

        model.fit(train_X, train_y)

        if test_size > 0:
            score = model.score(test_X, test_y)
            debug("model {} score: {}".format(name, score))
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame(model.feature_importances_, index=prepared.columns, columns=['importance'])
                importance = importance.sort_values('importance', ascending=False)
                located_low = []
                with open(IMPORTANCE_LOW_CURRENT, "w") as file:
                    for i, row in importance.iterrows():
                        if row.importance < IMPORTANCE_THRESHOLD:
                            file.write(i + "\n")
                            located_low.append(i)
                    if len(located_low) > 0:
                        debug("low importance features located: {}".format(located_low))

                with open(OUTPUT_PATH + "text/importance.txt", "w") as file:
                    file.write(str(importance))

        if name is not None:
            with open(MODEL_PATH.format(name), 'wb') as f:
                pickle.dump(model, f)

    def prepared_columns_diff(self):
        debug('columns diff')
        with open(COLUMNS_AFTER_PREPARE.format("train"), 'rb') as f:
            train_cols = pickle.load(f)
        with open(COLUMNS_AFTER_PREPARE.format("predict"), 'rb') as f:
            predict_cols = pickle.load(f)

        diff = set(train_cols).symmetric_difference(set(predict_cols))
        diff.remove(COL_Y)
        if len(diff) > 0:
            debug("mismatching columns located")
            debug(diff)

    def predict_submission(self, name):
        debug("predict start")
        with open(MODEL_PATH.format(name), 'rb') as f:
            model = pickle.load(f)
        prepared = self.prepare_data(suffix="predict")

        self.prepared_columns_diff()

        predictions = model.predict(prepared)
        debug(predictions)
        prices = [10 ** p for p in predictions]
        submission = pd.DataFrame(data={'SalePrice': prices})
        submission.index.names = ['Id']
        submission.index += 1461
        submission.to_csv(SUBMISSION_PATH, header=True, index=True)

    def catplot_per_category(self, df, category, prefix=""):
        debug("catplot for {}".format(category))
        fig = sns.catplot(x=COL_Y, y=category, kind="violin", inner="stick", data=df, height=5).fig
        fig.savefig(OUTPUT_PATH + "graph/{}catplot_{}.png".format(prefix, category))
        plt.close(fig)

    def pairplot_for_numeric_column(self, df, c):
        debug("pairplot for {}".format(c))
        fig = sns.pairplot(df[[c, COL_Y]], markers='+', height=5, diag_kws=dict(bins=30)).fig
        fig.savefig(OUTPUT_PATH + "graph/pair_plot_{}.png".format(c))
        plt.close(fig)

    def catplot_for_categories(self):
        for c in self.categoric_columns:
            self.catplot_per_category(self.df, c)

    def pairplot_for_numeric(self):
        for c in self.numeric_columns:
            if c != COL_Y:
                self.pairplot_for_numeric_column(self.df, c)

    def model_hyperspace(self, model, hyperspace_params):
        df = self.prepare_data(suffix="predict")

        debug("splitting")
        df = df.reindex(np.random.RandomState(seed=SEED).permutation(df.index))
        y = np.array(df[COL_Y])
        df = df.drop(COL_Y, 1)

        X = np.array(df)
        search = RandomizedSearchCV(estimator=model,
                                    param_distributions=hyperspace_params,
                                    cv=4,
                                    verbose=10,
                                    random_state=SEED,
                                    n_iter=100,
                                    n_jobs=-1)
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
        x = self.df
        all_data_na = (x.isnull().sum() / len(x)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        with open(OUTPUT_PATH + "text/missing_values.txt", "w") as file:
            file.write(str(missing_data))

        fig, ax = plt.subplots(figsize=(15, 12))
        plt.xticks(rotation='90')
        sns.barplot(x=all_data_na.index, y=all_data_na)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        fig.savefig(OUTPUT_PATH + "graph/missing_values.png")

    def remove_low_importance(self, model):
        open(IMPORTANCE_LOW_TOTAL, "w").close()

        while True:
            self.train_model(None, model)
            with open(IMPORTANCE_LOW_CURRENT) as file:
                cols = file.read().splitlines()
            if len(cols) == 0:
                debug("no more low importance features")
                return
            with open(IMPORTANCE_LOW_TOTAL) as file:
                existing = file.read().splitlines()
            existing.extend(cols)
            existing.sort()
            with open(IMPORTANCE_LOW_TOTAL, "w") as f:
                for c in existing:
                    f.write("{}\n".format(c))


pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 10000)
pd.set_option('display.float_format', lambda x: '%.6f' % x)
rcParams.update({'figure.autolayout': True})

h = House()
# h.general()
# h.missing_values()
# h.catplot_for_categories()
# h.pairplot_for_numeric()
# h.random_forest_hyperspace()

classifier_names = []
regressors = []
classifier_names.append("random_forest")
regressors.append(RandomForestRegressor(n_estimators=800,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        max_features='sqrt',
                                        max_depth=100,
                                        bootstrap=False,
                                        random_state=SEED))

classifier_names.append("gradient_boosting")
regressors.append(GradientBoostingRegressor(n_estimators=3000,
                                            learning_rate=0.05,
                                            max_depth=4,
                                            max_features='sqrt',
                                            min_samples_leaf=15,
                                            min_samples_split=10,
                                            loss='huber',
                                            random_state=SEED))

# h.remove_low_importance(regressor)

averaged = AveragingModels(classifier_names, regressors)
classifier_name = "average"
h.train_model(classifier_name, averaged)

h.train_model(classifier_name, averaged, test_size=0)
h = House(data_path=TEST_PATH)
h.predict_submission(classifier_name)
