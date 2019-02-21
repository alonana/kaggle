import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

TRAIN_PATH = '../data/train.csv'

df = pd.read_csv(TRAIN_PATH)
print(df.head())
print(df.describe())

for c in ['Survived', 'Pclass', "SibSp", 'Parch', 'Fare']:
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    df[c].plot.hist(ax=ax)
    plt.title(c)
    fig.savefig("../output/{}_hist.png".format(c))
