import pathlib
from datetime import datetime

import pandas as pd

DATA_FOLDER = "../data"
TRAIN_DATA = "%s/train.csv" % DATA_FOLDER
BREED_LABELS = "%s/breed_labels.csv" % DATA_FOLDER
COLOR_LABELS = "%s/color_labels.csv" % DATA_FOLDER

NA = 'N/A'


def debug(msg, new_line=False):
    separator = ''
    if new_line:
        separator = '\n'
    print("{} ===> {}{}".format(datetime.now(), separator, msg))


class Pets:
    def __init__(self, calculation_limit_rows=None):
        pathlib.Path("../output").mkdir(parents=True, exist_ok=True)
        self.calculation_limit_rows = calculation_limit_rows
        self.df = pd.read_csv(TRAIN_DATA, nrows=self.calculation_limit_rows)

    def pre_process(self):
        breeds = self.load_breeds()
        colors = self.load_colors()
        yes_no_not_sure = {
            1: 'Yes',
            2: 'No',
            3: NA,
        }
        self.translate_column('Type', {1: 'dog', 2: 'cat'})
        self.df.rename(columns={'Age': 'AgeMonths'}, inplace=True)
        self.translate_column('Breed1', breeds)
        self.translate_column('Breed2', breeds)
        self.translate_column('Gender', {1: 'Male', 2: 'Female', 3: 'Mixed'})
        self.translate_column('Color1', colors)
        self.translate_column('Color2', colors)
        self.translate_column('Color3', colors)
        self.translate_column('MaturitySize', {0: NA, 1: 'Small', 2: 'Medium', 3: 'Large', 4: 'ExtraLarge'})
        self.translate_column('FurLength', {0: NA, 1: 'Short', 2: 'Medium', 3: 'Long'})
        self.translate_column('Vaccinated', yes_no_not_sure)
        self.translate_column('Dewormed', yes_no_not_sure)
        self.translate_column('Sterilized', yes_no_not_sure)
        self.translate_column('Health', {0: NA, 1: 'Healthy', 2: 'Minor Injury', 3: 'Serious Injury'})
        debug(self.df.head(self.calculation_limit_rows), new_line=True)

    def translate_column(self, column, translation):
        self.df[column] = self.df.apply(lambda row: self.produce_translation(row, column, translation), axis=1)

    def produce_translation(self, row, column, translation):
        v = row[column]
        if v in translation:
            return translation[v]
        raise Exception('column {} has unexpected value {}'.format(column, v))

    def load_breeds(self):
        breeds_data = pd.read_csv(BREED_LABELS)
        breeds = {
            0: NA,
        }
        for i, row in breeds_data.iterrows():
            breeds[row['BreedID']] = row['BreedName']
        return breeds

    def load_colors(self):
        colors_data = pd.read_csv(COLOR_LABELS)
        colors = {
            0: NA,
        }
        for i, row in colors_data.iterrows():
            colors[row['ColorID']] = row['ColorName']
        return colors


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 100000)

p = Pets(calculation_limit_rows=10)
p.pre_process()
