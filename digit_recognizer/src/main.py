import math
import os
import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import rotate, zoom
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.models import Sequential

TRAIN_PATH = '../data/train.csv'
AUGMENT_PATH = '../data/augment.csv'
MODEL_PATH = '../output/model.h5'
WRONGS_PATH = '../output/wrongs.txt'
IMAGE_SIZE = 28
NUM_CLASSES = 10


def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


class DigitRecognizer:

    def __init__(self, path=TRAIN_PATH):
        debug("loading csv from {} starting".format(path))
        self.data_frame = pd.read_csv(path)
        debug("loading csv done, loaded {} rows".format(self.data_frame.shape[0]))

    def save_csv(self, path):
        debug("save csv to {}".format(path))
        self.data_frame.to_csv(path)

    def display_sample(self):
        debug("display sample")
        num_images = self.data_frame.shape[0]
        display_amount = 25
        random_rows = np.random.randint(0, num_images - 1, display_amount)
        self.display_images_by_index(random_rows)

    def augmentate_sample(self, amount, display=False):
        debug('augmentation for {} samples'.format(amount))
        indexes = [random.randint(0, self.data_frame.shape[0]) for i in range(amount)]
        display_indexes = []
        for i in indexes:
            display_indexes.append(i)
            display_indexes.append(self.data_frame.shape[0])
            self.add_augmentate(i)
        if display:
            self.display_images_by_index(display_indexes)

    def add_augmentate(self, i):
        debug("augment index {}".format(i))
        angle = random.randint(-30, 30)
        new_row = self.data_frame.iloc[i].copy()
        image_matrix = new_row.values[1:].reshape(IMAGE_SIZE, IMAGE_SIZE)
        image_matrix = rotate(image_matrix, angle, reshape=False)

        image_matrix = self.random_zoom(image_matrix)

        points_array = image_matrix.reshape(IMAGE_SIZE * IMAGE_SIZE)
        new_row[1:] = points_array
        self.data_frame = self.data_frame.append(new_row)

    def random_zoom(self, image_matrix):
        h, w = image_matrix.shape[:2]
        zoom_factor = 1 - random.uniform(0, 0.3)
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = np.zeros_like(image_matrix)
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (image_matrix.ndim - 2)
        out[top:top + zh, left:left + zw] = zoom(image_matrix, zoom_tuple)
        image_matrix = out
        return image_matrix

    def display_images_by_index(self, row_indexes, labels_suffix=None):
        debug("display indexes {}".format(row_indexes))
        fig = plt.figure(figsize=(IMAGE_SIZE, IMAGE_SIZE))
        plt.subplots_adjust(hspace=0.35)
        figures_per_edge = math.ceil(math.sqrt(len(row_indexes)))
        ax = []
        for image_index, row_index in enumerate(row_indexes):
            row = self.data_frame.iloc[row_index]
            image_matrix = self.data_frame.iloc[row_index, 1:].values.reshape(IMAGE_SIZE, IMAGE_SIZE)
            subplot = fig.add_subplot(figures_per_edge, figures_per_edge, image_index + 1)
            label = row['label']
            if labels_suffix is not None:
                label = "{} {}".format(label, labels_suffix[image_index])
            subplot.set_title(label)
            ax.append(subplot)
            plt.axis('off')
            plt.imshow(image_matrix, cmap="gray")
        plt.show()

    def data_prep(self):
        debug("data prep starting")
        out_y = keras.utils.to_categorical(self.data_frame.label, NUM_CLASSES)

        num_images = self.data_frame.shape[0]
        x_as_array = self.data_frame.values[:, 1:]
        x_shaped_array = x_as_array.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        out_x = x_shaped_array / 255
        debug("data prep done")
        return out_x, out_y

    def build_model(self):
        debug("build model")

        x, y = self.data_prep()

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

    def predict(self):
        debug("predict start")
        model = keras.models.load_model(MODEL_PATH)
        print(model.summary())
        x, y = self.data_prep()

        debug("make predictions")
        predictions_probabilities = model.predict(x).tolist()
        debug("locate wrongs")
        predictions = self.get_predictions_from_probabilities(predictions_probabilities)
        labels = []
        for label in y.tolist():
            c = label.index(max(label))
            labels.append(c)
        debug(predictions)
        debug(labels)
        wrongs = []
        for index, label in enumerate(labels):
            if label != predictions[index]:
                wrongs.append((index, predictions_probabilities[index]))
        debug("{} wrongs: {}".format(len(wrongs), wrongs))
        with open(WRONGS_PATH, "wb") as fp:
            pickle.dump(wrongs, fp)

    @staticmethod
    def get_predictions_from_probabilities(predictions_probabilities):
        predictions = []
        for probabilities in predictions_probabilities:
            c = probabilities.index(max(probabilities))
            predictions.append(c)
        return predictions

    def display_wrongs(self):
        with open(WRONGS_PATH, "rb") as fp:
            wrongs = pickle.load(fp)
        debug(wrongs)
        wrongs_index = []
        wrongs_probabilities = []
        for index, probability in wrongs:
            wrongs_index.append(index)
            wrongs_probabilities.append(probability)
        predictions = self.get_predictions_from_probabilities(wrongs_probabilities)
        predictions = [" predict={}".format(prediction) for prediction in predictions]
        display_amount = 49
        self.display_images_by_index(wrongs_index[:display_amount], predictions[:display_amount])


debug("starting in folder {}".format(os.getcwd()))
dr = DigitRecognizer()
dr.display_sample()
dr.augmentate_sample(40000)
dr.save_csv(AUGMENT_PATH)
dr = DigitRecognizer(AUGMENT_PATH)
dr.build_model()
dr.predict()
# dr.display_wrongs()
debug("done")

# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 26, 26, 20)        200
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 24, 24, 20)        3620
# _________________________________________________________________
# flatten (Flatten)            (None, 11520)             0
# _________________________________________________________________
# dense (Dense)                (None, 128)               1474688
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1290
# =================================================================
# Epoch 1/2
# 33600/33600 [==============================] - 132s 4ms/step - loss: 0.1785 - acc: 0.9456 - val_loss: 0.0709 - val_acc: 0.9764
# Epoch 2/2
# 33600/33600 [==============================] - 146s 4ms/step - loss: 0.0506 - acc: 0.9841 - val_loss: 0.0476 - val_acc: 0.9852
# 2019-02-06 22:00:36.484171 ===> 338 wrongs


#
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 26, 26, 20)        200
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 24, 24, 20)        3620
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 22, 22, 20)        3620
# _________________________________________________________________
# flatten (Flatten)            (None, 9680)              0
# _________________________________________________________________
# dense (Dense)                (None, 128)               1239168
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1290
# =================================================================
# Epoch 1/2
# 33600/33600 [==============================] - 193s 6ms/step - loss: 0.1503 - acc: 0.9536 - val_loss: 0.0568 - val_acc: 0.9840
# Epoch 2/2
# 33600/33600 [==============================] - 189s 6ms/step - loss: 0.0480 - acc: 0.9855 - val_loss: 0.0566 - val_acc: 0.9806
# 2019-02-06 22:20:43.720672 ===> 426 wrongs
