import pathlib
from datetime import datetime
from os import listdir
from os.path import join

import pandas as pd
from PIL import Image
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.models import Sequential

IMAGE_SIZE = 20

DATA_PATH = '../data'
OUTPUT_PATH = '../output'
GRAY_SCALE_PATH = '{}/gray_scale'.format(OUTPUT_PATH)
MODEL_PATH = "{}/model.dat".format(OUTPUT_PATH)


# TODO; convert to grayscale
def debug(msg, new_line=False):
    separator = ''
    if new_line:
        separator = '\n'
    print("{} ===> {}{}".format(datetime.now(), separator, msg))


class StreetChars:
    def __init__(self, folder, scratch=False, train_mode=True):
        self.train_mode = train_mode
        pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        pathlib.Path(GRAY_SCALE_PATH).mkdir(parents=True, exist_ok=True)
        if scratch:
            self.convert_to_gray_scale(folder)
            self.create_input_dataframe_from_gray_scale(folder)
        path = '{}/{}.csv'.format(OUTPUT_PATH, folder)
        self.df = pd.read_csv(path, index_col=0)
        self.labels = pd.read_csv('{}/trainLabels.csv'.format(DATA_PATH), index_col=0)
        self.classes = list(self.labels['Class'].unique())
        self.classes.sort()
        self.number_of_classes = len(self.classes)
        if not train_mode:
            self.labels = None

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

    def create_input_dataframe_from_gray_scale(self, folder_name):
        debug("create inputs from images")
        folder_path = '{}/{}'.format(GRAY_SCALE_PATH, folder_name)
        images = []
        for f in listdir(folder_path):
            file_path = join(folder_path, f)
            image = Image.open(file_path)
            image_pixels = []
            for p in list(image.getdata()):
                image_pixels.append(p)
            images.append(image_pixels)
        debug("loaded {} images, each contains {} inputs".format(len(images), len((images[0]))))
        debug("first row {}".format(images[0]))
        df = pd.DataFrame(images)
        df.to_csv('{}/{}.csv'.format(OUTPUT_PATH, folder_name))

    def data_prepare(self):
        debug("data prepare starting")

        num_images = self.df.shape[0]

        x_as_array = self.df.values
        x_shaped_array = x_as_array.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        out_x = x_shaped_array / 255

        if self.train_mode:
            out_y = pd.get_dummies(self.labels['Class']).values
        else:
            out_y = None

        debug("data prepare done")
        return out_x, out_y

    def build_model(self, epochs=10):
        x, y = self.data_prepare()
        debug("build model")

        model = Sequential()
        model.add(Conv2D(IMAGE_SIZE,
                         kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
        model.add(Conv2D(IMAGE_SIZE, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.number_of_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(x, y,
                  batch_size=32,
                  epochs=epochs,
                  validation_split=0.2)

        model.save(MODEL_PATH)

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


charsTrain = StreetChars("trainResized", scratch=False)
# charsTrain.build_model()
charsTrain.predict()

# charsPredict = StreetChars("testResized", scratch=False, train_mode=False)
