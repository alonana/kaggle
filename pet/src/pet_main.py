import pathlib
from datetime import datetime

import pandas as pd

TRAIN_DATA = "../data/train.csv"


def debug(msg, new_line=False):
    separator = ''
    if new_line:
        separator = '\n'
    print("{} ===> {}{}".format(datetime.now(), separator, msg))


class Pets:
    def __init__(self, calculation_limit_rows=None):
        pathlib.Path("../output").mkdir(parents=True, exist_ok=True)
        self.calculation_limit_rows = calculation_limit_rows

    def pre_process(self):
        df = pd.read_csv(TRAIN_DATA, nrows=self.calculation_limit_rows)
        df['Type'] = df.apply(lambda row: 'dog' if row['Type'] == 1 else 'cat' if row['Type'] == 2 else None, axis=1)
        df.rename(columns={'Age': 'AgeMonths'}, inplace=True)
        debug(df.head(), new_line=True)


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 100000)

p = Pets(calculation_limit_rows=10)
p.pre_process()
