import datetime
import pathlib
import re

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


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
        df.to_csv("../output/clean.csv")


b = MyBag()
b.clean_data()
