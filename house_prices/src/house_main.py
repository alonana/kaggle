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
from scipy.stats import skew
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
SKEWED_HANDLING_COLUMNS = OUTPUT_PATH + "text/skewed_columns.txt"
USE_COLUMNS_LIST = OUTPUT_PATH + "text/use_columns_list.txt"

MODEL_PATH = OUTPUT_PATH + 'model/{}.dat'
SUBMISSION_PATH = OUTPUT_PATH + 'my_submission.csv'

SEED = 42


def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


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
        numeric_columns = self.df._get_numeric_data().columns
        self.categoric_columns = list(set(self.df.columns) - set(numeric_columns))

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
        x['total_sf'] = x['totalbsmtsf'].fillna(0) + x['1stflrsf'].fillna(0) + x['2ndflrsf'].fillna(0)
        x['total_sqr_footage'] = (x['bsmtfinsf1'].fillna(0) + x['bsmtfinsf2'].fillna(0) +
                                  x['1stflrsf'].fillna(0) + x['2ndflrsf'].fillna(0))
        x['total_bathrooms'] = (x['fullbath'].fillna(0) + (0.5 * x['halfbath'].fillna(0)) +
                                x['bsmtfullbath'].fillna(0) + (0.5 * x['bsmthalfbath'].fillna(0)))
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

        # for c in self.continuous_numbers:
        #     self.create_normalized_columns(x, c)
        return x

    def handle_use_columns(self, x, create_columns_list):
        columns = []
        for c in x:
            columns.append(c)
        if create_columns_list:
            with open(USE_COLUMNS_LIST, 'wb') as f:
                pickle.dump(columns, f)
        else:
            with open(USE_COLUMNS_LIST, 'rb') as f:
                use_columns = pickle.load(f)
            for c in columns:
                if c not in use_columns:
                    x.drop(c, axis=1, inplace=True)

        return x

    def prepare_data(self, create_columns_list, suffix=''):
        x = self.df.copy()

        x = self.fill_missing_values(x)
        self.display_missing_values(x)

        for c in self.discrete_numbers:
            x[c] = x[c].astype('category')

        # x = self.replace_skew(x)

        # for c in self.continuous_numbers:
        #     x.drop(c, axis=1, inplace=True)

        x = pd.get_dummies(x)

        x = self.handle_use_columns(x, create_columns_list)

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
        prepared = self.prepare_data(True, suffix="train")

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
        prepared = self.prepare_data(False, suffix="predict")

        self.prepared_columns_diff()

        predictions = model.predict(prepared)
        debug(predictions)
        prices = [10 ** p for p in predictions]
        submission = pd.DataFrame(data={'SalePrice': prices})
        submission.index.names = ['Id']
        submission.index += 1461
        submission.to_csv(SUBMISSION_PATH, header=True, index=True)

    def catplot_per_category(self, df, category):
        debug("catplot for {}".format(category))
        fig = sns.catplot(x=COL_Y, y=category, kind="violin", inner="stick", data=df, height=5).fig
        fig.savefig(OUTPUT_PATH + "graph/{}_catplot.png".format(category))
        plt.close(fig)

    def pairplot_for_numeric_column(self, df, c):
        debug("pairplot for {}".format(c))
        fig = sns.pairplot(df[[c, COL_Y]], markers='+', height=5, diag_kws=dict(bins=30)).fig
        fig.savefig(OUTPUT_PATH + "graph/{}_pairplot.png".format(c))
        plt.close(fig)

    def catplot_for_categories(self):
        for c in self.categoric_columns:
            self.catplot_per_category(self.df, c)

    def pairplot_for_numeric(self):
        for c in self.df._get_numeric_data().columns:
            if c != COL_Y:
                self.pairplot_for_numeric_column(self.df, c)

    def model_hyperspace(self, model, hyperspace_params):
        df = self.prepare_data(True, suffix="predict")

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

    def fill_missing_values(self, df):
        x = df
        for c in ['poolqc', 'miscfeature', 'alley', 'fence', 'fireplacequ', 'garagecond', 'garagequal', 'garagefinish',
                  'garagetype', 'bsmtexposure', 'bsmtcond', 'bsmtqual', 'masvnrtype', 'electrical']:
            x[c] = x[c].fillna('None')

        for c in ['lotfrontage', 'garageyrblt', 'masvnrarea', 'bsmthalfbath', 'bsmtfullbath', 'garagearea', 'bsmtunfsf',
                  'bsmtfinsf1', 'bsmtfinsf2', 'totalbsmtsf', 'garagecars']:
            x[c] = x[c].fillna(0)

        x['mszoning'] = x['mszoning'].fillna('RL')
        x['functional'] = x['functional'].fillna('Typ')
        x['utilities'] = x['utilities'].fillna('AllPub')
        x['saletype'] = x['saletype'].fillna('Oth')
        x['kitchenqual'] = x['kitchenqual'].fillna('TA')

        x['garagecars'] = x['garagecars'].astype('int64')
        x['bsmtfullbath'] = x['bsmtfullbath'].astype('int64')

        return x

    def display_missing_values(self, x):
        all_data_na = (x.isnull().sum() / len(x)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        with open(OUTPUT_PATH + "text/missing_values.txt", "w") as file:
            file.write(str(missing_data))

        if all_data_na.size > 0:
            debug("Missing values located!!")
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

    def find_skew(self):
        numeric_feats = self.df.dtypes[self.df.dtypes != "object"].index
        numeric_feats = list(filter(lambda c: self.include_in_skew(c), numeric_feats))
        skewed_feats = self.df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew': skewed_feats})
        skewness = skewness[abs(skewness.Skew) > 0.75]
        debug("Handling Skew for {}".format(skewness.head(100)))
        columns = [c for c in skewness.index]
        with open(SKEWED_HANDLING_COLUMNS, 'wb') as f:
            pickle.dump(columns, f)

    def replace_skew(self, df):
        with open(SKEWED_HANDLING_COLUMNS, 'rb') as f:
            columns = pickle.load(f)
        from scipy.special import boxcox1p
        for c in columns:
            df["{}_skewed_fix".format(c)] = boxcox1p(df[c], 0.15)
            df.drop(c, axis=1, inplace=True)
        return df

    def include_in_skew(self, c):
        return 'join' not in c and 'haspool' not in c and not 'has' in c and not 'halfbath' in c


def get_regressor_random_forest():
    return "random_forest", RandomForestRegressor(n_estimators=800,
                                                  min_samples_split=2,
                                                  min_samples_leaf=1,
                                                  max_features='sqrt',
                                                  max_depth=100,
                                                  bootstrap=False,
                                                  random_state=SEED)


def get_regressor_gradient_boosting():
    return "gradient_boosting", GradientBoostingRegressor(n_estimators=3000,
                                                          learning_rate=0.05,
                                                          max_depth=4,
                                                          max_features='sqrt',
                                                          min_samples_leaf=15,
                                                          min_samples_split=10,
                                                          loss='huber',
                                                          random_state=SEED)


def get_regressor_averaged():
    classifier_names = []
    regressors = []
    for n, r in [get_regressor_random_forest(), get_regressor_gradient_boosting()]:
        classifier_names.append(n)
        regressors.append(r)

    averaged = AveragingModels(classifier_names, regressors)
    return "average", averaged


pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 10000)
pd.set_option('display.float_format', lambda x: '%.6f' % x)
rcParams.update({'figure.autolayout': True})

h = House()

# h.general()

# h.catplot_for_categories()
# h.pairplot_for_numeric()

# h.random_forest_hyperspace()

classifier_name, regressor = get_regressor_random_forest()

h.remove_low_importance(regressor)
# h.find_skew()

h.train_model(classifier_name, regressor)

# h.train_model(classifier_name, regressor, test_size=0)
# h_predict = House(data_path=TEST_PATH)
# h_predict.predict_submission(classifier_name)
