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

PREPARED_CSV = "../output/prepared.csv"
CLEAN_CSV = "../output/clean.csv"


def debug(msg, new_line=False):
    separator = ''
    if new_line:
        separator = '\n'
    print("{} ===> {}{}".format(datetime.now(), separator, msg))


class MyBag:

    def __init__(self, scratch=False):
        if scratch:
            nltk.download('stopwords')
        pathlib.Path("../output").mkdir(parents=True, exist_ok=True)
        self.cleaned_rows = 0

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

    def clean_data(self):
        df = pd.read_csv("../data/labeledTrainData.tsv", delimiter='\t', header=0, quoting=3)
        df.review = df.review.apply(self.clean_column)
        df.to_csv(CLEAN_CSV)

    def prepare_data(self):
        df = pd.read_csv(CLEAN_CSV)
        debug(df.head(), new_line=True)

        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                     max_features=5000)

        features = vectorizer.fit_transform(df.review)
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

    def train(self):
        debug("read prepared data")
        df = pd.read_csv(PREPARED_CSV)
        df.drop("Unnamed: 0", axis=1, inplace=True)
        debug(df.head(), new_line=True)
        model = RandomForestClassifier(n_estimators=100, verbose=1)
        y = np.array(df[COL_Y])
        df = df.drop(COL_Y, 1)

        X = np.array(df)
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
        model.fit(train_x, train_y)

        score = model.score(test_x, test_y)
        debug("model score: {}".format(score))
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame(model.feature_importances_, index=df.columns, columns=['importance'])
            importance = importance.sort_values('importance', ascending=False)
            with open("../output/importance.txt", "w") as file:
                file.write(str(importance))
        with open("../output/model.dat", 'wb') as f:
            pickle.dump(model, f)


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 100000)

b = MyBag()
# b.clean_data()
# b.prepare_data()
b.train()
