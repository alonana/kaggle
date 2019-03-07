import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

TEST_PATH = '../data/test.csv'
TRAIN_PATH = '../data/train.csv'
MODEL_PATH = '../output/model_{}.dat'
SUBMISSION_PATH = '../output/my_submission.csv'

SEED = 42


def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


class Titanic:

    def __init__(self):
        self.df = pd.read_csv(TRAIN_PATH)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        self.use_columns = ['Sex', 'Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']
        self.use_columns_numeric = ['Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']

    def general(self):
        debug("general")
        with open("../output/head.txt", "w") as file:
            file.write(str(self.df.head()))
        with open("../output/desc.txt", "w") as file:
            file.write(str(self.df.describe()))

    def hist(self):
        for c in self.use_columns_numeric:
            self.hist_for_column(c)

    def hist_for_column(self, c):
        debug("hist for {}".format(c))
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.df[c].plot.hist(ax=ax, bins=20)
        plt.title(c)
        fig.savefig("../output/hist_{}.png".format(c))

    def pair_plot(self):
        debug("pair plot")
        df2 = self.df.drop('PassengerId', 1).drop('Name', 1)
        sns.pairplot(df2).fig.savefig("../output/pair_plot.png")

    def hex(self):
        for c in self.use_columns_numeric:
            self.hex_for_column(c)

    def hex_for_column(self, c):
        debug("hex for column {}".format(c))
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.df.plot.hexbin(x='Survived', y=c, ax=ax)
        plt.title(c)
        fig.savefig("../output/hex_Survived_{}.png".format(c))

    def two_flavors_hist(self):
        for c in self.use_columns:
            self.two_flavors_hist_for_column(c)

    def two_flavors_hist_for_column(self, c):
        if c in ['Sex', 'Embarked']:
            debug("two flavor bar for {}".format(c))
            fig, ax = plt.subplots()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            self.df.groupby([c, 'Survived'])[c].count().unstack().plot.bar(ax=ax, legend=True)
            plt.title(c)
            plt.legend(['Died', 'Survived'])
            fig.savefig("../output/two_flavor_bar_{}.png".format(c))
        else:
            debug("two flavor hist for {}".format(c))
            fig, ax = plt.subplots()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            x = self.df
            bins = 10
            if c in ['Age', 'Fare']:
                bins = 40
            if c == 'Fare':
                selection = x['Fare'] < 100
                x = x[selection]
            x.groupby(['Survived'])[c].plot.hist(ax=ax, alpha=0.5, bins=bins)
            plt.title(c)
            plt.legend(['Died', 'Survived'])
            fig.savefig("../output/two_flavor_hist_{}.png".format(c))

    def prepare_data(self, df):
        x = df

        x['Pclass'] = x['Pclass'].astype('category')

        x['Female'] = np.where(x['Sex'] == 'female', 1, 0)

        x['CabinFirst'] = x['Cabin'].astype(str).str[0]
        x['CabinBCDEF'] = np.where(
            (x['CabinFirst'] == 'B') |
            (x['CabinFirst'] == 'C') |
            (x['CabinFirst'] == 'D') |
            (x['CabinFirst'] == 'E') |
            (x['CabinFirst'] == 'F'), 1, 0)

        x['Age0_6'] = np.where(x['Age'] < 7, 1, 0)
        x['Age12_15'] = np.where((x['Age'] >= 12) & (x['Age'] <= 15), 1, 0)
        x['Age32_35'] = np.where((x['Age'] >= 32) & (x['Age'] <= 35), 1, 0)
        x['Age52_53'] = np.where((x['Age'] >= 52) & (x['Age'] <= 53), 1, 0)
        x['Age75'] = np.where(x['Age'] > 75, 1, 0)

        x['Fare30'] = np.where(x['Fare'] < 30, 1, 0)

        x['SibSp0'] = np.where(x['SibSp'] == 0, 1, 0)
        x['SibSp1'] = np.where(x['SibSp'] == 1, 1, 0)
        x['SibSp2'] = np.where(x['SibSp'] == 2, 1, 0)
        x['SibOther'] = np.where(x['SibSp'] > 2, 1, 0)

        x['Parch0'] = np.where(x['Parch'] == 0, 1, 0)
        x['Parch1'] = np.where(x['Parch'] == 1, 1, 0)
        x['Parch2'] = np.where(x['Parch'] == 2, 1, 0)
        x['Parch3'] = np.where(x['Parch'] == 3, 1, 0)
        x['ParchOther'] = np.where(x['Parch'] > 3, 1, 0)

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

        x = pd.get_dummies(x)

        print(x.head(10))
        return x

    def investigate(self):
        x = self.df
        # x['CabinFirst'] = x['Cabin'].astype(str).str[0]
        # print(x.head(5))
        fig, ax = plt.subplots()
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # x.groupby(['Survived'])['CabinFirst'].plot.hist(ax=ax, alpha=0.5)
        x.groupby(['Embarked', 'Survived']).CabinFirst.count().unstack().plot.bar(ax=ax, legend=True)
        plt.title("investigate")
        plt.legend(['Died', 'Survived'])
        fig.savefig("../output/investigate.png")

    def check_predictions(self, name):
        debug("check predict start")
        with open(MODEL_PATH.format(name), 'rb') as f:
            model = pickle.load(f)
        prepared = self.prepare_data(self.df)

        predictions = model.predict(prepared.drop('Survived', 1))

        correct = prepared[prepared['Survived'] == predictions]
        total = prepared.shape[0]
        amount = correct.shape[0]
        debug("correct {} / {} = {}\n{}".format(amount, total, amount / total, correct.head()))

        wrong = prepared[prepared['Survived'] != predictions]
        total = prepared.shape[0]
        amount = wrong.shape[0]
        debug("wrong {} / {} = {}\n{}".format(amount, total, amount / total, wrong.head(100)))

    def run_model(self, name, model):
        df_relevant = self.prepare_data(self.df)

        debug("splitting")
        df_relevant = df_relevant.reindex(np.random.RandomState(seed=SEED).permutation(df_relevant.index))
        y = np.array(df_relevant['Survived'])
        df_relevant = df_relevant.drop('Survived', 1)

        X = np.array(df_relevant)
        train_X, test_X, train_y, test_y = \
            train_test_split(X, y, test_size=0.3, random_state=SEED)

        debug("checking mode {}".format(name))
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)
        errors = abs(predictions - test_y)
        errors = np.mean(errors)
        debug('Mean Absolute Error for {} is {}'.format(name, errors))
        with open(MODEL_PATH.format(name), 'wb') as f:
            pickle.dump(model, f)

        return errors

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
                        error = t.run_model(key, SVC(kernel=kernel, gamma=gamma, C=c))
                        errors[key] = error

        for key in sorted(errors, key=errors.get, reverse=True):
            print("{} : {}".format(key, errors[key]))


t = Titanic()
# t.general()
# t.hist()
# t.hex()
# t.pair_plot()
# t.two_flavors_hist()
# t.run_model("random_forest", RandomForestClassifier(n_estimators=1000, random_state=SEED))
# t.run_model("gnb", GaussianNB())
# t.run_model("knn", KNeighborsClassifier())
# t.run_model("mnb", MultinomialNB())
# t.run_model("bnb", BernoulliNB())
# t.run_model("lr", LogisticRegression())
# t.run_model("sdg", SGDClassifier())
# t.run_model("lsvc", LinearSVC())
# t.run_model("nsvc", NuSVC())
# t.run_model("nsvc", SVC())
# t.svc_tune()
# t.investigate()
# t.predict_submission("random_forest")
t.check_predictions("random_forest")
