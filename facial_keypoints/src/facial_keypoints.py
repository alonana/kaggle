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
LOOKUP_PATH = '../data/IdLookupTable.csv'
OUTPUT_PATH = '../output'
IMAGES_PATH = "{}/images".format(OUTPUT_PATH)
MODELS_PATH = "{}/models".format(OUTPUT_PATH)
PREDICTIONS_PATH = '{}/predictions'.format(OUTPUT_PATH)
SUBMISSION_PATH = '{}/submission.csv'.format(OUTPUT_PATH)
IMAGE_SIZE = 96


def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


class Facial:
    def __init__(self, path=TRAIN_PATH, max_rows=None, train_mode=True, target_labels_index=None):
        self.train_mode = train_mode
        self.target_labels = []
        self.build_target_labels()
        if target_labels_index is not None:
            self.target_labels = self.target_labels[target_labels_index * 2:target_labels_index * 2 + 2]
            self.name = self.target_labels[0]
        else:
            self.name = 'all'

        debug(self.target_labels)
        pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        pathlib.Path("{}/{}".format(IMAGES_PATH, self.name)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
        pathlib.Path(PREDICTIONS_PATH).mkdir(parents=True, exist_ok=True)
        debug("loading csv from {} starting".format(path))
        self.df = pd.read_csv(path, nrows=max_rows)
        debug("loading csv done, loaded {} rows".format(self.df.shape[0]))
        if train_mode and target_labels_index is not None:
            for l in self.target_labels:
                self.df = self.df.dropna(subset=[l])
            debug("after empty values cleanup {} rows".format(self.df.shape[0]))

    def model_path(self):
        return "{}/{}.dat".format(MODELS_PATH, self.name)

    def predictions_path(self, name_override=None):
        if name_override is None:
            use = self.name
        else:
            use = name_override

        return "{}/{}.csv".format(PREDICTIONS_PATH, use)

    def build_target_labels(self):
        bilateral = ['DDD_eye_center', 'DDD_eye_inner_corner', 'DDD_eye_outer_corner', 'DDD_eyebrow_inner_end',
                     'DDD_eyebrow_outer_end', 'mouth_DDD_corner']

        for n in bilateral:
            self.add_target_label(n.replace("DDD", "left"))
            self.add_target_label(n.replace("DDD", "right"))

        for n in ['nose_tip', 'mouth_center_top_lip', 'mouth_center_bottom_lip']:
            self.add_target_label(n)

    def add_target_label(self, name):
        self.target_labels += ['{}_x'.format(name), '{}_y'.format(name)]

    def save_images(self, train_mode=True):
        if train_mode:
            predictions = None
        else:
            predictions = pd.read_csv(self.predictions_path(), dtype={'Index': str},
                                      index_col='Index')

        for i, row in self.df.iterrows():
            pixels = self.get_pixels(row)
            image = np.reshape(pixels, (IMAGE_SIZE, IMAGE_SIZE))
            plt.imsave("{}/{}/{}.png".format(IMAGES_PATH, self.name, i), image)
            labels_iter = iter(self.target_labels)

            if train_mode:
                points_row = row
            else:
                points_row = predictions.iloc[i]

            for x_label, y_label in zip(labels_iter, labels_iter):
                x = int(points_row[x_label])
                y = int(points_row[y_label])
                image[y][x] = 255
            plt.imsave("{}/{}/{}_points.png".format(IMAGES_PATH, self.name, i), image)

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

    def build_model(self, epochs=500):
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
                  epochs=epochs,
                  validation_split=0.2)

        model.save(self.model_path())

    def predict(self):
        debug("predict start")
        model = keras.models.load_model(self.model_path())
        x, y = self.data_prepare()

        predictions = model.predict(x).tolist()
        index = ["row{}".format(x + 1) for x in range(len(predictions))]
        predictions_df = pd.DataFrame(predictions, columns=self.target_labels, index=index)
        predictions_df.index.name = 'Index'
        predictions_df.to_csv(self.predictions_path())
        debug("predict done")

    def submit(self, max_rows=None):
        debug("submit starting")

        joined_predictions = None
        for target_index in range(0, len(self.target_labels), 2):
            print(joined_predictions)
            label = self.target_labels[target_index]
            predictions = pd.read_csv(self.predictions_path(label), dtype={'Index': str}, index_col='Index')
            if joined_predictions is None:
                joined_predictions = predictions
            else:
                joined_predictions = pd.merge(joined_predictions, predictions, on='Index')

        lookup = pd.read_csv(LOOKUP_PATH, nrows=max_rows, dtype={'ImageId': str, 'RowId': str}, index_col='RowId')
        for i, row in lookup.iterrows():
            image = row['ImageId']
            label = row['FeatureName']
            key = "row{}".format(image)
            prediction_row = joined_predictions.loc[key]
            location = prediction_row[label]
            location = max(0, location)
            location = min(96, location)
            lookup.loc[i, 'Location'] = location

        lookup[["Location"]].to_csv(SUBMISSION_PATH)
        debug("submit done")

    def display_missing_values(self):
        all_data_na = (self.df.isnull().sum() / len(self.df)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        print(str(missing_data))


# f = Facial()
# f.display_missing_values()


for target_labels_index in range(15):
    train_rows = None
    predict_rows = None
    epochs = 500

    # f = Facial(max_rows=train_rows, target_labels_index=target_labels_index)
    # f.build_model(epochs=epochs)

    f = Facial(path=TEST_PATH, train_mode=False, max_rows=predict_rows, target_labels_index=target_labels_index)
    f.predict()
    # f.save_images(train_mode=False)

f = Facial(path=TEST_PATH, train_mode=False, max_rows=1)
f.submit()
