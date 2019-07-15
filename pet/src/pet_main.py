import json
import pathlib
import pickle
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

OUTPUT_FOLDER = "../output/"
GRAPHS_FOLDER = "%s/graphs/" % OUTPUT_FOLDER
DATA_FOLDER = "../data"
MODEL_PATH = "%s/model.dat" % OUTPUT_FOLDER
CLEAN_WORDS = "%s/words_clean_{}.csv" % OUTPUT_FOLDER
WORDS_FEATURES = "%s/words_features_{}.csv" % OUTPUT_FOLDER
WORDS_VECTORIZER = "%s/words_vectorizer.dat" % OUTPUT_FOLDER
TRAIN_DATA = "%s/train.csv" % DATA_FOLDER
TEST_DATA = "%s/test.csv" % DATA_FOLDER
BREED_LABELS = "%s/breed_labels.csv" % DATA_FOLDER
COLOR_LABELS = "%s/color_labels.csv" % DATA_FOLDER
STATE_LABELS = "%s/state_labels.csv" % DATA_FOLDER
TRAIN_COLUMNS = "%s/train_columns.json" % DATA_FOLDER
SUBMISSION_PATH = "%s/submission.csv" % DATA_FOLDER

IMPORTANCE_THRESHOLD = 0.001
IMPORTANCE_LOW_CURRENT = OUTPUT_FOLDER + "importance_low_current.txt"
IMPORTANCE_LOW_TOTAL = OUTPUT_FOLDER + "importance_low_total.txt"

COL_ID = 'PetID'
COL_Y = 'AdoptionSpeed'
SEED = 42
WORDS_FEATURES_LIMIT = 1000

NA = 'N/A'


def debug(msg, new_line=False):
    separator = ''
    if new_line:
        separator = '\n'
    print("{} ===> {}{}".format(datetime.now(), separator, msg))


class Pets:
    def __init__(self, csv_path, train_mode, calculation_limit_rows=None):
        pathlib.Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        pathlib.Path(GRAPHS_FOLDER).mkdir(parents=True, exist_ok=True)
        self.train_mode = train_mode
        self.calculation_limit_rows = calculation_limit_rows
        if calculation_limit_rows:
            self.head_lines = calculation_limit_rows
        else:
            self.head_lines = 5
        self.df = pd.read_csv(csv_path, nrows=self.calculation_limit_rows)
        debug("{} rows loaded".format(self.df.shape[0]))
        self.pre_process()
        self.numeric_columns = self.df._get_numeric_data().columns
        self.categoric_columns = list(set(self.df.columns) - set(self.numeric_columns))
        self.cleaned_rows = 0

    def words_clean_column(self, text):
        text = str(text)
        clean = re.sub("[^A-Za-z]", " ", text)
        clean = clean.lower()
        words = clean.split()
        words = [w for w in words if w not in stopwords.words('english')]
        result = ' '.join(words)
        self.cleaned_rows += 1
        if self.cleaned_rows % 300 == 0:
            debug("cleaned {} rows".format(self.cleaned_rows))
        return result

    def get_clean_word_path(self):
        return CLEAN_WORDS.format("train" if self.train_mode else "predict")

    def get_word_features_path(self):
        return WORDS_FEATURES.format("train" if self.train_mode else "predict")

    def bag_of_words_clean(self):
        debug('clean_words')
        self.cleaned_rows = 0
        bag = pd.DataFrame()
        bag['text'] = self.df['Description'].apply(self.words_clean_column)
        bag.to_csv(self.get_clean_word_path())

    def bag_of_words_prepare(self):
        df = pd.read_csv(self.get_clean_word_path(), nrows=self.calculation_limit_rows)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        debug("preparing words from text:\n{}".format(df.head()))
        df['text'] = df.apply(lambda row: "" if row['text'] is np.nan else row['text'], axis=1)

        vectorizer = CountVectorizer(analyzer="word",
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=WORDS_FEATURES_LIMIT)

        features = vectorizer.fit_transform(df.text)

        with open(WORDS_VECTORIZER, 'wb') as f:
            pickle.dump(vectorizer, f)

        features = features.toarray()
        features_names = vectorizer.get_feature_names()
        amounts = np.sum(features, axis=0)
        distribution = []
        for w, c in zip(features_names, amounts):
            distribution.append((w, c))
        distribution.sort(key=lambda x: x[1], reverse=True)
        debug("distribution: {}".format(distribution))
        prepared = pd.DataFrame(columns=features_names, data=features)
        prepared = prepared.applymap(lambda x: 0 if x == 0 else 1)
        debug("words features:\n{}".format(prepared.head()))
        prepared.to_csv(self.get_word_features_path())

    def pre_process(self):
        debug('pre process')
        states = self.load_states()
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
        self.translate_column('State', states)
        debug("table post pre process:\n{}".format(self.df.head(self.head_lines)))

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

    def load_states(self):
        states_data = pd.read_csv(STATE_LABELS)
        states = {
            0: NA,
        }
        for i, row in states_data.iterrows():
            states[row['StateID']] = row['StateName']
        return states

    def display_missing_values(self):
        debug('analyze missing values')
        all_data_na = (self.df.isnull().sum() / len(self.df)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        with open(OUTPUT_FOLDER + "/missing_values.txt", "w") as file:
            file.write(str(missing_data))

        if all_data_na.size > 0:
            debug("Missing values located!!")
            fig, ax = plt.subplots(figsize=(15, 12))
            plt.xticks(rotation='90')
            sns.barplot(x=all_data_na.index, y=all_data_na)
            plt.xlabel('Features', fontsize=15)
            plt.ylabel('Percent of missing values', fontsize=15)
            plt.title('Percent missing data by feature', fontsize=15)
            fig.savefig(OUTPUT_FOLDER + "/missing_values.png")

    def prepare_data(self):
        debug('prepare data')
        # debug (self.df.info())
        # self.display_missing_values()

        words = pd.read_csv(self.get_word_features_path())
        words.drop('Unnamed: 0', axis=1, inplace=True)
        new_words_names = {}
        for n in words.columns:
            new_words_names[n] = 'word_' + n
        words.rename(columns=new_words_names, inplace=True)
        debug("words features:\n{}".format(words.head()))

        self.df = pd.merge(self.df, words, left_on=None, right_on=None, left_index=True, right_index=True)

        self.df.drop('Name', axis=1, inplace=True)
        self.df.drop('RescuerID', axis=1, inplace=True)
        self.df.drop('Description', axis=1, inplace=True)
        self.df.drop(COL_ID, axis=1, inplace=True)
        self.df = pd.get_dummies(self.df)

        with open(IMPORTANCE_LOW_TOTAL) as file:
            for c in file.read().splitlines():
                if c in self.df:
                    self.df.drop(c, axis=1, inplace=True)

        if self.train_mode:
            with open(TRAIN_COLUMNS, 'w') as f:
                json.dump(list(self.df.columns), f)
        debug("prepared data:\n{}".format(self.df.head(self.head_lines)))

    def train_model(self, test_size=0.3):
        self.prepare_data()

        debug('train model')
        model = RandomForestClassifier(n_estimators=1000,
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       max_features='sqrt',
                                       max_depth=100,
                                       bootstrap=False,
                                       verbose=10,
                                       random_state=SEED)

        y = np.array(self.df[COL_Y])
        prepared = self.df.drop(COL_Y, 1)

        x = np.array(prepared)
        train_x, test_x, train_y, test_y = \
            train_test_split(x, y, test_size=test_size, random_state=SEED)

        model.fit(train_x, train_y)

        if test_size > 0:
            score = model.score(test_x, test_y)
            debug("model score: {}".format(score))
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

                with open(OUTPUT_FOLDER + "importance.txt", "w") as file:
                    file.write(str(importance))

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

    def align_with_train_columns(self):
        with open(TRAIN_COLUMNS, 'r') as f:
            train_columns = json.load(f)
        missing_columns = set(train_columns) - set(self.df.columns) - {COL_Y}
        redundant_columns = set(self.df.columns) - set(train_columns)
        debug("missing columns: {}".format(missing_columns))
        debug("redundant columns: {}".format(redundant_columns))
        for c in missing_columns:
            self.df[c] = 0
        for c in redundant_columns:
            self.df.drop(c, axis=1, inplace=True)

    def predict_prepare_words(self):
        with open(WORDS_VECTORIZER, 'rb') as f:
            vectorizer = pickle.load(f)

        clean_words = pd.read_csv(self.get_clean_word_path(), nrows=self.calculation_limit_rows)
        clean_words.drop("Unnamed: 0", axis=1, inplace=True)
        clean_words['text'] = clean_words.apply(lambda row: "" if row['text'] is np.nan else row['text'], axis=1)
        debug("predict words:\n{}".format(clean_words.head()))

        words_features = vectorizer.transform(clean_words.text)
        words_features = words_features.toarray()
        features_names = vectorizer.get_feature_names()
        prepared = pd.DataFrame(columns=features_names, data=words_features)
        prepared = prepared.applymap(lambda x: 0 if x == 0 else 1)
        debug("words features:\n{}".format(prepared.head()))
        prepared.to_csv(self.get_word_features_path())

    def predict(self):
        self.prepare_data()
        self.align_with_train_columns()
        x = np.array(self.df)

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict(x)
        debug(predictions)

        data = pd.DataFrame(predictions, columns=[COL_Y])
        data[COL_ID] = pd.read_csv(TEST_DATA, nrows=self.calculation_limit_rows)[COL_ID]
        data = data[[COL_ID, COL_Y]]
        data.set_index(COL_ID, inplace=True)
        debug(data.head(), new_line=True)
        data.to_csv(SUBMISSION_PATH)

    def catplot_per_category(self, df, category):
        debug("catplot for {}".format(category))
        fig = sns.catplot(x=COL_Y, y=category, kind="violin", inner="stick", data=df, height=5).fig
        fig.savefig(GRAPHS_FOLDER + "{}_catplot.png".format(category))
        plt.close(fig)

    def pairplot_for_numeric_column(self, df, c):
        debug("pairplot for {}".format(c))
        fig = sns.pairplot(df[[c, COL_Y]], markers='+', height=5, diag_kws=dict(bins=30)).fig
        fig.savefig(GRAPHS_FOLDER + "{}_pairplot.png".format(c))
        plt.close(fig)

    def boxplot_for_numeric_column(self, df, c):
        debug("boxplot for {}".format(c))
        fig, ax = plt.subplots()
        sns.boxplot(x=COL_Y, y=c, data=df, ax=ax)
        plt.savefig(GRAPHS_FOLDER + "{}_boxplot.png".format(c))
        plt.close(fig)

    def boxplot_for_category_column(self, df, c):
        debug("boxplot for {}".format(c))
        fig, ax = plt.subplots()
        sns.boxplot(x=c, y=COL_Y, data=df, ax=ax)
        plt.savefig(GRAPHS_FOLDER + "{}_boxplot.png".format(c))
        plt.close(fig)

    def catplot_for_categories(self):
        for c in self.categoric_columns:
            self.catplot_per_category(self.df, c)

    def boxplot_for_categories(self):
        for c in self.categoric_columns:
            if c != 'Name' and c != 'RescuerID' and c != 'Description' and c != 'PetID':
                self.boxplot_for_category_column(self.df, c)

    def pairplot_for_numeric(self):
        for c in self.numeric_columns:
            if c != COL_Y:
                self.pairplot_for_numeric_column(self.df, c)

    def boxplot_for_numeric(self):
        for c in self.numeric_columns:
            if c != COL_Y:
                self.boxplot_for_numeric_column(self.df, c)


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 100000)

train = Pets(TRAIN_DATA, True, calculation_limit_rows=None)
# train.bag_of_words_clean()
# train.bag_of_words_prepare()
# train.catplot_for_categories()
# train.pairplot_for_numeric()
train.boxplot_for_numeric()
train.boxplot_for_categories()
# train.train_model()

# predict = Pets(TEST_DATA, False, calculation_limit_rows=None)
# predict.bag_of_words_clean()
# predict.predict_prepare_words()
# predict.predict()
