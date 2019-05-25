import pathlib
from datetime import datetime
from os import listdir
from os.path import join

import pandas as pd
from PIL import Image

IMAGE_SIZE = 20

DATA_PATH = '../data'
OUTPUT_PATH = '../output'
GRAY_SCALE_PATH = '{}/gray_scale'.format(OUTPUT_PATH)


# TODO; convert to grayscale
def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


class StreetChars:
    def __init__(self, folder, scratch=False, train_mode=True):
        self.train_mode = train_mode
        pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        pathlib.Path(GRAY_SCALE_PATH).mkdir(parents=True, exist_ok=True)
        if scratch:
            self.convert_to_gray_scale(folder)
            self.create_input_dataframe_from_gray_scale(folder)
        self.df = pd.read_csv('{}/{}.csv'.format(OUTPUT_PATH, folder))
        self.data_prepare()

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
        x_shaped_array = x_as_array.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
        out_x = x_shaped_array / 255

        if self.train_mode:
            labels = pd.read_csv('{}/trainLabels.csv'.format(DATA_PATH))
            out_y = labels.values[:, 1:1]
        else:
            out_y = None

        debug("data prepare done")
        return out_x, out_y


chars = StreetChars("trainResized", scratch=False)
