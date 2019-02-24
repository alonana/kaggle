from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

TRAIN_PATH = '../data/train.csv'


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
        self.df[c].plot.hist(ax=ax)
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

    def random_forest(self):
        # https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
        df_relevant = self.df.drop('PassengerId', 1).drop('Name', 1).drop('Ticket', 1).drop('Cabin', 1)
        df_relevant = pd.get_dummies(df_relevant)
        print(df_relevant.head())
        y = np.array(df_relevant['Survived'])
        df_relevant = df_relevant.drop('Survived', 1)
        X = np.array(df_relevant)
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
        rf = RandomForestRegressor(n_estimators=1000, random_state=42)
        rf.fit(train_X, train_y)
        predictions = rf.predict(test_X)
        errors = abs(predictions - test_y)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


t = Titanic()
# t.general()
# t.hist()
# t.hex()
# t.pair_plot()
# t.two_flavors_hist()
t.random_forest()
