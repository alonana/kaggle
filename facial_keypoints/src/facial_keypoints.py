import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_PATH = '../data/training.csv'
TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../output'
IMAGE_SIZE = 96


def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


class Facial:
    def __init__(self, path=TRAIN_PATH, max_rows=None):
        pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        debug("loading csv from {} starting".format(path))
        self.df = pd.read_csv(path, nrows=max_rows)
        debug("loading csv done, loaded {} rows".format(self.df.shape[0]))

    def save_images(self):
        pixels = [int(x) for x in self.df.iloc[0]['Image'].split()]
        debug(pixels)
        image = np.reshape(pixels, (IMAGE_SIZE, IMAGE_SIZE))
        debug(image)
        plt.imsave("{}/i.png".format(OUTPUT_PATH), pixels)


f = Facial(max_rows=1)
# debug (f.df.head())
f.save_images()
