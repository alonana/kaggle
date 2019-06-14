import pathlib
import pickle
import re
from datetime import datetime
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from augmentation import combined_random
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

IMAGE_SIZE = 20
COUNT_THRESHOLD = 0

DATA_PATH = '../data'
OUTPUT_PATH = '../output'
CLASSES_PATH = '{}/classes.dat'.format(OUTPUT_PATH)
GRAY_SCALE_PATH = '{}/gray_scale'.format(OUTPUT_PATH)
MODEL_PATH = "{}/model.dat".format(OUTPUT_PATH)


def debug(msg, new_line=False):
    separator = ''
    if new_line:
        separator = '\n'
    print("{} ===> {}{}".format(datetime.now(), separator, msg))


class StreetChars:
    def __init__(self, folder, scratch=False, train_mode=True, augment=False):
        self.train_mode = train_mode
        self.augment = augment
        pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        pathlib.Path(GRAY_SCALE_PATH).mkdir(parents=True, exist_ok=True)
        if scratch:
            self.convert_to_gray_scale(folder)
            self.create_input_dataframe_from_gray_scale(folder)
        path = '{}/{}.csv'.format(OUTPUT_PATH, folder)
        self.df = pd.read_csv(path, index_col=0)
        if train_mode:
            self.labels = pd.read_csv('{}/trainLabels.csv'.format(DATA_PATH), index_col=0)

            self.remove_infrequent()

            self.classes = list(self.labels['Class'].unique())
            self.classes.sort()
            self.number_of_classes = len(self.classes)
            with open(CLASSES_PATH, "wb") as fp:
                pickle.dump(self.classes, fp)
            debug("{} classes located: {}".format(len(self.classes), self.classes))
        else:
            self.labels = None
            with open(CLASSES_PATH, "rb") as fp:
                self.classes = pickle.load(fp)
        head_size = 2
        debug("row example\n{}".format(self.df.head(head_size)))
        if self.labels is not None:
            debug("labels example\n{}".format(self.labels.head(head_size)))

    def remove_infrequent(self):
        infrequent = self.print_samples_per_class()
        remove_rows = []
        for i, row in self.labels.iterrows():
            if row['Class'] in infrequent:
                remove_rows.append(i - 1)

        self.labels.drop(self.labels.index[remove_rows], inplace=True)
        self.df.drop(self.df.index[remove_rows], inplace=True)
        debug("{} rows after removal".format(self.df.shape[0]))
        self.print_samples_per_class()

    def print_samples_per_class(self):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 5000)
        counts = self.labels.Class.value_counts()
        # debug("frequencies:\n{}".format(counts))
        return [c for c in counts[counts < COUNT_THRESHOLD].index]

    def convert_to_gray_scale(self, folder_name):
        debug("covert to gray scale")
        input_folder = '{}/{}'.format(DATA_PATH, folder_name)
        output_folder = '{}/{}'.format(GRAY_SCALE_PATH, folder_name)
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        for f in listdir(input_folder):
            file_path = join(input_folder, f)
            image = Image.open(file_path)
            image = image.convert('L')
            image.save('{}/{}'.format(output_folder, f))

    def tryint(self, s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(self, s):
        # Turn a string into a list of string and number chunks.            "z23a" -> ["z", 23, "a"]
        return [self.tryint(c) for c in re.split('([0-9]+)', s)]

    def create_input_dataframe_from_gray_scale(self, folder_name):
        debug("create inputs from images")
        folder_path = '{}/{}'.format(GRAY_SCALE_PATH, folder_name)
        images = []
        sorted_files = listdir(folder_path)
        sorted_files.sort(key=self.alphanum_key)
        for f in sorted_files:
            file_path = join(folder_path, f)
            image = Image.open(file_path)
            image_pixels = [f]
            for p in list(image.getdata()):
                image_pixels.append(p)
            images.append(image_pixels)
        debug("loaded {} images, each contains {} inputs".format(len(images), len((images[0]))))
        df = pd.DataFrame(images)
        df.to_csv('{}/{}.csv'.format(OUTPUT_PATH, folder_name))

    def data_prepare(self):
        debug("data prepare starting")

        num_images = self.df.shape[0]

        x_as_array = self.df.iloc[:, 1:].values
        x_shaped_array = x_as_array.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
        if self.augment:
            for i in range(num_images):
                x_shaped_array[i] = combined_random(x_shaped_array[i])
        x_shaped_array = x_shaped_array.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)

        out_x = x_shaped_array / 255

        if self.train_mode:
            out_y = pd.get_dummies(self.labels['Class']).values
        else:
            out_y = None

        debug("data prepare done")
        return out_x, out_y

    def create_layers(self):
        model = Sequential()

        model.add(layers.Conv2D(32,
                                kernel_size=(3, 3),
                                activation='relu',
                                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))

        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64,
                         kernel_size=(3, 3),
                         activation='relu'))

        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128,
                         kernel_size=(3, 3),
                         activation='relu'))

        model.add(layers.Flatten())

        model.add(layers.Dense(256, activation='relu'))

        model.add(layers.Dense(self.number_of_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])
        debug(model.summary())
        return model

    def build_model(self, epochs=10):
        debug("build model")

        model = self.create_layers()

        val_acc = []
        acc = []
        for i in range(epochs):
            debug("epoch {}".format(i))
            x, y = self.data_prepare()
            history = model.fit(x, y,
                                batch_size=32,
                                epochs=1,
                                validation_split=0.2)
            acc.append(history.history['acc'])
            val_acc.append(history.history['val_acc'])
            self.save_accuracy_graph(acc, val_acc)

        model.save(MODEL_PATH)

    def save_accuracy_graph(self, acc, val_acc):
        fig, ax = plt.subplots(figsize=(15, 12))
        plt.plot(acc, color='blue', label='acc')
        plt.plot(val_acc, color='red', label='val_acc')
        plt.legend()
        fig.savefig(OUTPUT_PATH + "/accuracy.png")
        plt.close(fig)

    def get_predictions_from_probabilities(self, predictions_probabilities):
        predictions = []
        for probabilities in predictions_probabilities:
            c = probabilities.index(max(probabilities))
            predictions.append(c)
        return predictions

    def predict(self):
        debug("predict start")
        model = keras.models.load_model(MODEL_PATH)
        x, y = self.data_prepare()

        debug("make predictions")
        predictions_probabilities = model.predict(x).tolist()
        predictions = self.get_predictions_from_probabilities(predictions_probabilities)
        predictions_classes = [self.classes[x] for x in predictions]
        return predictions_classes, y

    def predict_train(self):
        predictions_classes, y = self.predict()
        labels = []
        for label in y.tolist():
            c = label.index(max(label))
            labels.append(c)
        labels_classes = [self.classes[x] for x in labels]
        debug("predictions: {}".format(predictions_classes))
        debug("labels     : {}".format(labels_classes))

        debug("locate wrongs")
        wrongs = []
        for index, label in enumerate(labels_classes):
            file = self.df.iloc[index][0]
            predicted = predictions_classes[index]
            if label != predicted:
                wrongs.append("{}={} !{}".format(file, label, predicted))
        percent = 100 * len(wrongs) // len(labels_classes)
        debug("{}% = {} wrongs out of {}: {}".format(percent, len(wrongs), len(labels_classes), wrongs))

    def submit(self):
        debug("submit start")
        predictions_classes, y = self.predict()
        debug("predictions: {}".format(predictions_classes))
        submission = pd.read_csv('{}/sampleSubmission.csv'.format(DATA_PATH), index_col=0)
        submission.drop("Class", inplace=True, axis=1)
        submission["Class"] = predictions_classes
        submission.to_csv('{}/submission.csv'.format(OUTPUT_PATH))


# charsTrain = StreetChars("trainResized", scratch=False,augment=True)
# charsTrain.build_model(epochs=100)
# charsTrain.predict_train()

charsPredict = StreetChars("testResized", scratch=True, train_mode=False)
charsPredict.submit()
