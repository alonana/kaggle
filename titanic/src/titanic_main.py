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


def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


class Titanic:

    def __init__(self):
        self.df = pd.read_csv(TRAIN_PATH)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        self.use_columns = ['Sex', 'Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']
        self.use_columns_numeric = ['Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']
        self.train_X = None
        self.test_X = None
        self.train_y = None
        self.test_y = None

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
        for c in self.use_columns_numeric:
            self.two_flavors_hist_for_column(c)

    def two_flavors_hist_for_column(self, c):
        debug("two flavor hist for {}".format(c))
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.df.groupby(['Survived'])[c].plot.hist(ax=ax, alpha=0.5)
        plt.title(c)
        plt.legend(['Died', 'Survived'])
        fig.savefig("../output/two_flavor_hist_{}.png".format(c))

    def prepare_data(self, df):
        x = df

        x['Pclass'] = x['Pclass'].astype('category')

        x['CabinFirst'] = x['Cabin'].astype(str).str[0]

        x = df.drop('PassengerId', 1).drop('Name', 1).drop('Ticket', 1).drop('Cabin', 1)

        x['AgeIndication'] = np.where(x['Age'].isnull(), 1, 0)
        x['Age'] = np.where(x['Age'].isnull(), x['Age'].mean(), x['Age'])
        x['Age'] = (x['Age'] - x['Age'].mean()) / x['Age'].std()

        x['Fare'] = (x['Fare'] - x['Fare'].mean()) / x['Fare'].std()
        x = pd.get_dummies(x)

        if 'CabinFirst_T' in x:
            x = x.drop('CabinFirst_T', 1)

        x = x.drop("Sex_male", 1)

        # print(x.head(n=5))
        return x

    def prepare_and_split(self):
        if self.train_X is None:
            df_relevant = self.prepare_data(self.df)

            debug("splitting")
            df_relevant = df_relevant.reindex(np.random.permutation(df_relevant.index))
            y = np.array(df_relevant['Survived'])
            df_relevant = df_relevant.drop('Survived', 1)
            X = np.array(df_relevant)
            self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=0.3,
                                                                                    random_state=42)
        return self.train_X, self.test_X, self.train_y, self.test_y

    def run_model(self, name, model):
        train_X, test_X, train_y, test_y = self.prepare_and_split()

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

        fare = prepared['Fare']
        mean = fare.mean()
        prepared['Fare'] = fare.fillna(mean)

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
# t.run_model("random_forest", RandomForestClassifier(n_estimators=1000))
# t.run_model("gnb", GaussianNB())
# t.run_model("knn", KNeighborsClassifier())
# t.run_model("mnb", MultinomialNB())
# t.run_model("bnb", BernoulliNB())
# t.run_model("lr", LogisticRegression())
# t.run_model("sdg", SGDClassifier())
# t.run_model("lsvc", LinearSVC())
# t.run_model("nsvc", NuSVC())
# t.svc_tune()
t.predict_submission("svc_rbf_gamma_0.1_c_1")
