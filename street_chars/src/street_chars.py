import pathlib
from datetime import datetime
from os import listdir
from os.path import join

from PIL import Image

DATA_PATH = '../data'
OUTPUT_PATH = '../output'


# TODO; convert to grayscale
def debug(msg):
    print("{} ===> {}".format(datetime.now(), msg))


class StreetChars:
    def __init__(self, scratch=False):
        pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        if scratch:
            self.create_input_dataframe_from_images("trainResized")

    def create_input_dataframe_from_images(self, folder_name):
        debug("create inputs for folder {}".format(folder_name))
        images = []
        folder_path = '{}/{}'.format(DATA_PATH, folder_name)
        for f in listdir(folder_path):
            file_path = join(folder_path, f)
            image = Image.open(file_path)
            image_pixels = []
            for p in list(image.getdata()):
                image_pixels.append(p[0])
                image_pixels.append(p[1])
                image_pixels.append(p[2])
            images.append(image_pixels)
        debug("loaded {} images, each contains {} inputs".format(len(images), len((images[0]))))


chars = StreetChars(scratch=True)
