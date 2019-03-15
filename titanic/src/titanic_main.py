import io
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

TEST_PATH = '../data/test.csv'
TRAIN_PATH = '../data/train.csv'
MODEL_PATH = '../model/{}.dat'
SUBMISSION_PATH = '../output/my_submission.csv'

SEED = 42


def debug(msg):
    print("{} ===>\n{}".format(datetime.now(), msg))


class Titanic:

    def __init__(self):
        self.df = pd.read_csv(TRAIN_PATH)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        rcParams.update({'figure.autolayout': True})
        self.use_columns = ['Sex', 'Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']
        self.use_columns_numeric = ['Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']

    def general(self):
        debug("general")

        with open("../output/general/head.txt", "w") as file:
            file.write(str(self.df.head()))

        with open("../output/general/desc.txt", "w") as file:
            file.write(str(self.df.describe()))

        with open("../output/general/desc_objects.txt", "w") as file:
            file.write(str(self.df.describe(include='O')))

        buf = io.StringIO()
        self.df.info(buf=buf)
        with open("../output/general/info.txt", "w") as file:
            file.write(buf.getvalue())

        self.df.groupby('Pclass').mean().to_csv("../output/general/mean_by_pclass.csv")
        self.df.groupby(['Pclass', 'Sex']).mean().to_csv("../output/general/mean_by_pclass_and_sex.csv")

        group_by_age = pd.cut(self.df["Age"], np.arange(0, 90, 10))
        self.df.groupby(group_by_age).mean().to_csv("../output/general/mean_by_age_cut.csv")

    def add_cabin_occupy(self, df):
        cabins = {}
        for cabin, count in df.Cabin.value_counts().iteritems():
            cabins[cabin] = count

        name = 'CabinOccupy'
        df[name] = self.df.apply(lambda r: cabins.get(r.Cabin, None), axis=1)
        df[name] = df[name].astype('category')

        return df

        # self.two_flavors_hist_for_column(self.df,'CabinOccupy')

    def hist(self):
        for c in self.use_columns_numeric:
            self.hist_for_column(self.df, c)

    def hist_for_column(self, df, c, prefix=""):
        debug("hist for {}".format(c))
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        df[c].plot.hist(ax=ax, bins=20)
        plt.title(c)
        fig.savefig("../output/{}hist_{}.png".format(prefix, c))

    def pair_plot(self):
        debug("pair plot")
        df2 = self.df.drop('PassengerId', 1).drop('Name', 1)
        sns.pairplot(df2).fig.savefig("../output/pair_plot.png")

    def two_flavors_hist(self, df, prefix=""):
        for c in self.use_columns:
            self.two_flavors_hist_for_column(df, c, prefix)

    def two_flavors_hist_for_column(self, df, c, prefix=""):
        if c in ['Sex', 'Embarked', 'combined']:
            debug("two flavor bar for {}".format(c))
            fig, ax = plt.subplots()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            df.groupby([c, 'Survived'])[c].count().unstack().plot.bar(ax=ax, legend=True)
            plt.title(c)
            plt.legend(['Died', 'Survived'])
            fig.savefig("../output/{}two_flavor_bar_{}.png".format(prefix, c))
        else:
            debug("two flavor hist for {}".format(c))
            fig, ax = plt.subplots()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            x = df
            bins = 10
            if c in ['Age', 'Fare']:
                bins = 40
            if c == 'Fare':
                selection = x['Fare'] < 100
                x = x[selection]
            x.groupby(['Survived'])[c].plot.hist(ax=ax, alpha=0.5, bins=bins)
            plt.title(c)
            plt.legend(['Died', 'Survived'])
            fig.savefig("../output/{}two_flavor_hist_{}.png".format(prefix, c))

    def prepare_data(self, df):
        x = df.copy()
        x = self.add_cabin_occupy(x)
        debug(x.head(100))
        x = self.age_regression(x)
        debug(x.head(100))

        x['Female'] = np.where(x['Sex'] == 'female', 1, 0)

        x['CabinFirst'] = x['Cabin'].astype(str).str[0]
        x['CabinBCDEF'] = np.where(
            (x['CabinFirst'] == 'B') |
            (x['CabinFirst'] == 'C') |
            (x['CabinFirst'] == 'D') |
            (x['CabinFirst'] == 'E') |
            (x['CabinFirst'] == 'F'), 1, 0)

        x['Age0_6'] = np.where(x['Age'] < 7, 1, 0)
        x['Age7_11'] = np.where((x['Age'] >= 7) & (x['Age'] <= 11), 1, 0)
        x['Age12_15'] = np.where((x['Age'] >= 12) & (x['Age'] <= 15), 1, 0)
        x['Age32_35'] = np.where((x['Age'] >= 32) & (x['Age'] <= 35), 1, 0)
        x['Age52_60'] = np.where((x['Age'] >= 52) & (x['Age'] <= 60), 1, 0)
        x['Age61Up'] = np.where(x['Age'] >= 61, 1, 0)

        x['Fare0_7'] = np.where(x['Fare'] <= 7, 1, 0)
        x['Fare_8_20'] = np.where((x['Fare'] >= 8) & (x['Fare'] <= 20), 1, 0)
        x['Fare_21_40'] = np.where((x['Fare'] >= 21) & (x['Fare'] <= 40), 1, 0)
        x['Fare_41Up'] = np.where(x['Fare'] >= 41, 1, 0)

        x['SibSp0'] = np.where(x['SibSp'] == 0, 1, 0)
        x['SibSp1'] = np.where(x['SibSp'] == 1, 1, 0)
        x['SibSp2'] = np.where(x['SibSp'] == 2, 1, 0)
        x['SibOther'] = np.where(x['SibSp'] > 2, 1, 0)

        x['Parch0'] = np.where(x['Parch'] == 0, 1, 0)
        x['Parch1'] = np.where(x['Parch'] == 1, 1, 0)
        x['Parch2'] = np.where(x['Parch'] == 2, 1, 0)
        x['Parch3'] = np.where(x['Parch'] == 3, 1, 0)
        x['ParchOther'] = np.where(x['Parch'] > 3, 1, 0)

        x['MalePclass3'] = np.where((x['Pclass'] == 3) & (x['Sex'] == 'male'), 1, 0)
        x['FemalePclass1'] = np.where((x['Pclass'] == 1) & (x['Sex'] == 'female'), 1, 0)

        x['Title'] = x.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        values = ['Mr', 'Miss', 'Mrs', 'Master']
        x['Title'] = np.where(x['Title'].isin(values), x['Title'], 'Other')

        x['family'] = x['SibSp'] + x['Parch']
        x['family0'] = np.where(x['family'] == 0, 1, 0)
        x['family1_3'] = np.where((x['family'] >= 1) & (x['family'] <= 3), 1, 0)
        x['family4up'] = np.where(x['family'] >= 4, 1, 0)

        x = x.drop('family', 1)
        x = x.drop('PassengerId', 1)
        x = x.drop('Name', 1)
        x = x.drop('Sex', 1)
        x = x.drop('Ticket', 1)
        x = x.drop('Cabin', 1)
        x = x.drop('CabinFirst', 1)
        x = x.drop('Age', 1)
        x = x.drop('Fare', 1)
        x = x.drop('SibSp', 1)
        x = x.drop('Parch', 1)

        x = x.drop('Parch3', 1)
        x = x.drop('ParchOther', 1)
        x = x.drop('SibSp2', 1)

        x = pd.get_dummies(x)

        if 'CabinOccupy_4.0' in x:
            x = x.drop('CabinOccupy_4.0', 1)

        return x

    def hist_by_cabin_first(self):
        x = self.df
        x['CabinFirst'] = x['Cabin'].astype(str).str[0]
        fig, ax = plt.subplots()
        x.groupby(['CabinFirst', 'Survived']).CabinFirst.count().unstack().plot.bar(ax=ax, legend=True)
        plt.title("survived by cabin")
        plt.legend(['Died', 'Survived'])
        fig.savefig("../output/survived_by_cabin.png")

    def confusion_matrix_plot(self, predictions, prepared):
        cm = confusion_matrix(prepared['Survived'], predictions)
        debug("confusion matrix: {}".format(cm))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        labels = ['Died', 'Survived']
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        fig.savefig("../output/confusion_matrix.png")

    def analyze_predictions_false_negative(self, x):
        prefix = "FN_"
        # self.two_flavors_hist(x, prefix=prefix)
        # self.hist_for_column(x, "Age", prefix)
        y = self.df.copy()
        combined = "combined"
        x[combined] = x.apply(lambda row: "{}_{}_{}".format(row['Sex'], row['SibSp'], row['Parch']), axis=1)
        y[combined] = y.apply(lambda row: "{}_{}_{}".format(row['Sex'], row['SibSp'], row['Parch']), axis=1)
        self.two_flavors_hist_for_column(x, combined, "combined_fp_")
        self.two_flavors_hist_for_column(y, combined, "combined_all_")

    def analyze_predictions(self, predictions):
        x = self.df
        correct = x[x['Survived'] == predictions]
        total = x.shape[0]
        amount = correct.shape[0]
        debug("correct {} / {} = {}".format(amount, total, amount / total))
        wrong = x[x['Survived'] != predictions]
        total = x.shape[0]
        amount = wrong.shape[0]
        debug("wrong {} / {} = {}".format(amount, total, amount / total))
        false_negative = wrong[wrong['Survived'] == 1]
        origin_false_negative = self.df.ix[false_negative.index.values]
        debug("false negative {} = {}".format(false_negative.shape[0], origin_false_negative.shape[0]))
        self.analyze_predictions_false_negative(origin_false_negative)

    def check_predictions(self, name):
        debug("check predict start")
        with open(MODEL_PATH.format(name), 'rb') as f:
            model = pickle.load(f)
        prepared = self.prepare_data(self.df)

        predictions = model.predict(prepared.drop('Survived', 1))

        self.confusion_matrix_plot(predictions, prepared)
        self.analyze_predictions(predictions)

    def train_model(self, name, model, test_size=0.3):
        df_relevant = self.prepare_data(self.df)

        debug("splitting")
        df_relevant = df_relevant.reindex(np.random.RandomState(seed=SEED).permutation(df_relevant.index))
        y = np.array(df_relevant['Survived'])
        df_relevant = df_relevant.drop('Survived', 1)

        X = np.array(df_relevant)
        train_X, test_X, train_y, test_y = \
            train_test_split(X, y, test_size=test_size, random_state=SEED)

        debug("fit model {}".format(name))
        model.fit(train_X, train_y)

        if test_size > 0:
            score = model.score(test_X, test_y)
            debug("model {} score: {}".format(name, score))
            importance = pd.DataFrame(model.feature_importances_, index=df_relevant.columns, columns=['importance'])
            importance = importance.sort_values('importance', ascending=False)
            debug("importance {}".format(importance))
        else:
            score = None

        with open(MODEL_PATH.format(name), 'wb') as f:
            pickle.dump(model, f)

        return score

    def model_hyperspace(self, model, hyperspace_params):
        df_relevant = self.prepare_data(self.df)

        debug("splitting")
        df_relevant = df_relevant.reindex(np.random.RandomState(seed=SEED).permutation(df_relevant.index))
        y = np.array(df_relevant['Survived'])
        df_relevant = df_relevant.drop('Survived', 1)

        X = np.array(df_relevant)

        search = GridSearchCV(estimator=model, param_grid=hyperspace_params, cv=5, verbose=10)
        search.fit(X, y)
        debug(search.best_params_)

    def random_forest_hyperspace(self):
        hyperspace_params = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 5, 6, 7, 8],
            'criterion': ['gini', 'entropy']
        }
        self.model_hyperspace(RandomForestClassifier(random_state=SEED), hyperspace_params)
        # {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'auto', 'n_estimators': 100}

    def predict_submission(self, name):
        debug("predict start")
        with open(MODEL_PATH.format(name), 'rb') as f:
            rf = pickle.load(f)
        df = pd.read_csv(TEST_PATH)
        prepared = self.prepare_data(df)

        predictions = rf.predict(prepared)
        submission = pd.DataFrame(data={"Survived": predictions})
        submission.index.names = ['PassengerId']
        submission.index += 892
        submission.to_csv(SUBMISSION_PATH, header=True, index=True)

    def svc_tune(self):
        errors = {}
        # for kernel in ['linear', 'rbf', 'poly']:
        for kernel in ['rbf', 'poly']:
            # for gamma in [0.01, 0.1, 1, 10, 100]:
            for gamma in [0.001, 0.01, 0.1]:
                for c in [0.1, 1, 10, 100, 1000]:
                    if (kernel != 'poly' or gamma < 10) and \
                            (kernel != 'linear' or c != 1000) and \
                            (kernel != 'poly' or c < 100):
                        key = "svc_{}_gamma_{}_c_{}".format(kernel, gamma, c)
                        error = t.train_model(key, SVC(kernel=kernel, gamma=gamma, C=c))
                        errors[key] = error

        for key in sorted(errors, key=errors.get, reverse=True):
            print("{} : {}".format(key, errors[key]))

    def factor_plot(self, df, prefix=""):
        x = df
        fig, ax = plt.subplots()
        sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=x, ax=ax)
        plt.title("survived by cabin")
        plt.legend(['Died', 'Survived'])
        fig.savefig("../output/{}factor.png".format(prefix))

    def extract_title(self):
        x = self.df
        x['Title'] = x.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        values = ['Mr', 'Miss', 'Mrs', 'Master']
        x['Title'] = np.where(x['Title'].isin(values), x['Title'], 'Other')
        debug(x['Title'])

    def hist_per_category(self, df, category, column):
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        category_values = df[category].unique()
        plt.hist([df.loc[df[category] == y, column] for y in category_values], label=category_values, bins=11)
        ax.legend()
        plt.title("{} histogram by {}".format(column, category))

        fig.savefig("../output/hist_{}_per_{}.png".format(column, category))

    def extract_family(self):
        x = self.df
        x['family'] = x['SibSp'] + x['Parch']
        self.hist_per_category(x, 'Survived', 'family')

    def age_regression(self, df):
        debug('Age regression')
        x = df.copy()
        x = x.dropna(subset=['Age'], how='any')
        # debug(x.head())
        x['Pclass'] = x['Pclass'].astype('category')
        x['SibSp'] = x['SibSp'].astype('category')
        x['Parch'] = x['Parch'].astype('category')
        self.hist_per_category(x, 'Pclass', "Age")

        # debug("pair plot")
        # z = x.copy()[['Age','Fare']]
        # sns.pairplot(z).fig.savefig("../output/age_fare_pair_plot.png")

        y = x['Age']
        X = x[['Pclass', 'Sex', 'SibSp', 'Parch']]
        X = pd.get_dummies(X)

        # debug(X.head())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        regressor = RandomForestRegressor(n_estimators=200, random_state=SEED)
        regressor.fit(X_train, y_train)
        importance = pd.DataFrame(regressor.feature_importances_, index=X.columns, columns=['importance'])
        importance = importance.sort_values('importance', ascending=False)
        # debug("importance {}".format(importance))
        y_pred = regressor.predict(X_test)
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        full = df.copy()
        full['Pclass'] = full['Pclass'].astype('category')
        full['SibSp'] = full['SibSp'].astype('category')
        full['Parch'] = full['Parch'].astype('category')
        full = full[['Pclass', 'Sex', 'SibSp', 'Parch']]
        full = pd.get_dummies(full)
        full = full.drop('SibSp_8', 1)

        # debug(full.head())
        df['AgePredict'] = regressor.predict(full).astype('int')
        df['Age'] = np.where(df['Age'].isnull(), df['AgePredict'], df['Age'])
        return df



t = Titanic()
# t.general()
# t.hist()
# t.hex()
# t.pair_plot()
# t.two_flavors_hist(t.df)
# t.train_model("gnb", GaussianNB())
# t.train_model("knn", KNeighborsClassifier())
# t.train_model("mnb", MultinomialNB())
# t.train_model("bnb", BernoulliNB())
# t.train_model("lr", LogisticRegression())
# t.train_model("sdg", SGDClassifier())
# t.train_model("lsvc", LinearSVC())
# t.train_model("nsvc", NuSVC())
# t.train_model("nsvc", SVC())
# t.svc_tune()
# t.hist_by_cabin_first()
# t.factor_plot(t.df)
# t.extract_title()
# t.extract_family()

classifier = RandomForestClassifier(criterion='entropy',
                                    max_depth=5,
                                    max_features='auto',
                                    n_estimators=500,
                                    random_state=SEED)

t.train_model("random_forest", classifier, test_size=0.3)
# t.check_predictions("random_forest")


# t.random_forest_hyperspace()
t.train_model("random_forest", classifier, test_size=0)
t.predict_submission("random_forest")
