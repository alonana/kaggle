import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.models import Sequential

TRAIN_PATH = '../data/train.csv'
MODEL_PATH = '../output/model.h5'
IMAGE_SIZE = 28
NUM_CLASSES = 10


def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


def display_sample():
    rows = 25
    data_frame = pd.read_csv(TRAIN_PATH, nrows=rows)
    fig = plt.figure(figsize=(IMAGE_SIZE, IMAGE_SIZE))
    plt.subplots_adjust(hspace=0.35)
    figures_per_edge = math.ceil(math.sqrt(rows))
    ax = []
    for row_index in range(rows):
        row = data_frame.iloc[row_index]
        image_matrix = data_frame.iloc[row_index, 1:].values.reshape(IMAGE_SIZE, IMAGE_SIZE)
        subplot = fig.add_subplot(figures_per_edge, figures_per_edge, row_index + 1)
        subplot.set_title(row['label'])
        ax.append(subplot)
        plt.axis('off')
        plt.imshow(image_matrix, cmap="gray")
    plt.show()


def data_prep(data_frame):
    out_y = keras.utils.to_categorical(data_frame.label, NUM_CLASSES)

    num_images = data_frame.shape[0]
    x_as_array = data_frame.values[:, 1:]
    x_shaped_array = x_as_array.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y


def build_model():
    data_frame = pd.read_csv(TRAIN_PATH)

    x, y = data_prep(data_frame)

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
              batch_size=128,
              epochs=2,
              validation_split=0.2)

    model.save(MODEL_PATH)


debug("starting in folder {}".format(os.getcwd()))
display_sample()
build_model()
debug("done")
