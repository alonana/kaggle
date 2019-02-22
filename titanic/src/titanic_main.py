from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

TRAIN_PATH = '../data/train.csv'


def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


class Titanic:

    def __init__(self):
        self.df = pd.read_csv(TRAIN_PATH)
        self.use_columns = ['Sex', 'Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']
        self.use_columns_numeric = ['Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']

    def general(self):
        debug("general")
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
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


t = Titanic()
t.general()
t.hist()
t.hex()
t.pair_plot()
