import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.models import Sequential

TRAIN_PATH = '../data/training.csv'
TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../output'
IMAGES_PATH = "{}/images".format(OUTPUT_PATH)
MODEL_PATH = "{}/model.dat".format(OUTPUT_PATH)
IMAGE_SIZE = 96
NUM_CLASSES = 2


def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


class Facial:
    def __init__(self, path=TRAIN_PATH, max_rows=None):
        pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        pathlib.Path(IMAGES_PATH).mkdir(parents=True, exist_ok=True)
        debug("loading csv from {} starting".format(path))
        self.df = pd.read_csv(path, nrows=max_rows)
        debug("loading csv done, loaded {} rows".format(self.df.shape[0]))

    def save_images(self):
        for i, row in self.df.iterrows():
            pixels = self.get_pixels(row)
            image = np.reshape(pixels, (IMAGE_SIZE, IMAGE_SIZE))
            plt.imsave("{}/{}.png".format(IMAGES_PATH, i), image)
            left_eye_center_x = int(row['left_eye_center_x'])
            left_eye_center_y = int(row['left_eye_center_y'])
            image[left_eye_center_y][left_eye_center_x] = 255
            plt.imsave("{}/{}_left_eye.png".format(IMAGES_PATH, i), image)

    def get_pixels(self, row):
        pixels = [int(x) for x in row['Image'].split()]
        return pixels

    def data_prepare(self):
        debug("data prep starting")

        rows = []
        for i, row in self.df.iterrows():
            new_row = [row.left_eye_center_x, row.left_eye_center_y] + self.get_pixels(row)
            rows.append(new_row)

        df = pd.DataFrame(rows, columns=['eye_x', 'eye_y'])

        num_images = self.df.shape[0]
        x_as_array = self.df.values[:, start_column:]
        x_shaped_array = x_as_array.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        out_x = x_shaped_array / 255
        debug("data prep done")
        return out_x, out_y

    def build_model(self):
        debug("build model")

        x, y = self.data_prepare()

        model = Sequential()
        model.add(Conv2D(20,
                         kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
        model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(x, y,
                  batch_size=32,
                  epochs=2,
                  validation_split=0.2)

        model.save(MODEL_PATH)


# f = Facial(max_rows=10)
# f.save_images()
f = Facial()
