import pathlib
import pickle
import re
from datetime import datetime

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

COL_Y = 'sentiment'
TRAIN_DATA = "../data/labeledTrainData.tsv"
TEST_DATA = "../data/testData.tsv"
MODEL_DAT = "../output/model.dat"
PREPARED_CSV = "../output/prepared.csv"
TRAIN_CLEAN_DATA = "../output/train_clean.csv"
TEST_CLEAN_DATA = "../output/test_clean.csv"
VECTORIZER = "../output/vectorizer.dat"


def debug(msg, new_line=False):
    separator = ''
    if new_line:
        separator = '\n'
    print("{} ===> {}{}".format(datetime.now(), separator, msg))


class MyBag:

    def __init__(self, scratch=False, calculation_limit_rows=None):
        if scratch:
            nltk.download('stopwords')
        pathlib.Path("../output").mkdir(parents=True, exist_ok=True)
        self.cleaned_rows = 0
        self.calculation_limit_rows = calculation_limit_rows

    def clean_column(self, text):
        clean = BeautifulSoup(text, features="html.parser").get_text()
        clean = re.sub("[^A-Za-z]", " ", clean)
        clean = clean.lower()
        words = clean.split()
        words = [w for w in words if w not in stopwords.words('english')]
        result = ' '.join(words)
        self.cleaned_rows += 1
        if self.cleaned_rows % 300 == 0:
            debug("cleaned {} rows".format(self.cleaned_rows))
        return result

    def clean_data(self, input_file_name, output_file_name):
        self.cleaned_rows = 0
        debug("clean data from {} to {}".format(input_file_name, output_file_name))
        df = pd.read_csv(input_file_name, delimiter='\t', header=0, quoting=3, nrows=self.calculation_limit_rows)
        df.review = df.review.apply(self.clean_column)
        df.to_csv(output_file_name)

    def prepare_data(self):
        debug("preparing data")
        df = pd.read_csv(TRAIN_CLEAN_DATA, nrows=self.calculation_limit_rows)
        debug(df.head(), new_line=True)

        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                     max_features=5000)

        features = vectorizer.fit_transform(df.review)

        with open(VECTORIZER, 'wb') as f:
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
        prepared[COL_Y] = df[COL_Y]
        prepared = prepared[[COL_Y] + features_names]
        debug(prepared.head(), new_line=True)
        prepared.to_csv(PREPARED_CSV)

    def train(self, test_size=0.3):
        debug("train starting - read prepared data")
        df = pd.read_csv(PREPARED_CSV, nrows=self.calculation_limit_rows)
        df.drop("Unnamed: 0", axis=1, inplace=True)
        debug(df.head(), new_line=True)
        model = RandomForestClassifier(n_estimators=100, verbose=1)
        y = np.array(df[COL_Y])
        df = df.drop(COL_Y, 1)

        x = np.array(df)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)
        model.fit(train_x, train_y)

        if test_size > 0:
            score = model.score(test_x, test_y)
            debug("model score: {}".format(score))
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame(model.feature_importances_, index=df.columns, columns=['importance'])
            importance = importance.sort_values('importance', ascending=False)
            with open("../output/importance.txt", "w") as file:
                file.write(str(importance))
        with open(MODEL_DAT, 'wb') as f:
            pickle.dump(model, f)

    def predict(self):
        debug("predict start")
        with open(VECTORIZER, 'rb') as f:
            vectorizer = pickle.load(f)

        df = pd.read_csv(TEST_CLEAN_DATA, nrows=self.calculation_limit_rows)
        df.drop("Unnamed: 0", axis=1, inplace=True)
        debug(df.head(), new_line=True)

        features = vectorizer.transform(df.review)
        features = features.toarray()

        debug("predict features {}".format(features))
        with open(MODEL_DAT, 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict(features)

        df.drop('review', axis=1, inplace=True)
        df[COL_Y] = predictions
        df.to_csv("../output/submission.csv", index=False, quoting=3)


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 100000)

b = MyBag()
b.clean_data(TRAIN_DATA, TRAIN_CLEAN_DATA)
b.prepare_data()
b.train(test_size=0)

b.clean_data(TEST_DATA, TEST_CLEAN_DATA)
b.predict()
