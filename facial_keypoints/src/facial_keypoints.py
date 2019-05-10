import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
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
    def __init__(self, path=TRAIN_PATH, max_rows=None, train_mode=True):
        self.train_mode = train_mode
        self.target_labels = ['left_eye_center_x', 'left_eye_center_y']
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
        debug("data prepare starting")

        prepared_columns = []
        if self.train_mode:
            prepared_columns += self.target_labels

        rows = []
        for i, row in self.df.iterrows():
            if self.train_mode:
                new_row = [row[c] for c in prepared_columns] + self.get_pixels(row)
            else:
                new_row = self.get_pixels(row)
            rows.append(new_row)

        pixels_start_column = len(prepared_columns)
        for pixel_row in range(IMAGE_SIZE):
            for pixel_col in range(IMAGE_SIZE):
                prepared_columns.append("p{}_{}".format(pixel_row, pixel_col))

        df = pd.DataFrame(rows, columns=prepared_columns)

        num_images = df.shape[0]

        x_as_array = df.values[:, pixels_start_column:]
        x_shaped_array = x_as_array.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
        out_x = x_shaped_array / 255

        if self.train_mode:
            y_as_array = df.values[:, 0:pixels_start_column]
            out_y = y_as_array.reshape(num_images, pixels_start_column)
        else:
            out_y = None

        debug("data prepare done")
        return out_x, out_y

    def build_model(self):
        x, y = self.data_prepare()
        debug("build model")

        model = Sequential()
        model.add(Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(self.target_labels), activation='relu'))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mae', 'accuracy'])
        model.fit(x, y,
                  batch_size=128,
                  epochs=500,
                  validation_split=0.2)

        model.save(MODEL_PATH)

    def predict(self):
        debug("predict start")
        model = keras.models.load_model(MODEL_PATH)
        print(model.summary())
        x, y = self.data_prepare()

        predictions = model.predict(x).tolist()
        debug(predictions)

    def display_missing_values(self):
        all_data_na = (self.df.isnull().sum() / len(self.df)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        print(str(missing_data))


def train():
    f = Facial(max_rows=100)
    # f.display_missing_values()
    # f.save_images()
    f.build_model()


def predict():
    f = Facial(max_rows=2, path=TEST_PATH, train_mode=False)
    f.predict()


train()
# predict()
