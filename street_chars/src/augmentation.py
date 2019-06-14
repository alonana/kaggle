import pathlib
from random import randint, uniform

import numpy as np
from PIL import Image
from skimage import filters, transform

OUTPUT_PATH = "../output/augmentation"
ORIGINAL_PATH = '/home/alon/git/kaggle/street_chars/output/gray_scale/trainResized/1.Bmp'


def fast_warp(img, tf, output_shape, mode='wrap'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)


def same(pixels):
    return pixels


def warp(pixels):
    size = pixels.shape[0]
    # random rotations
    dorotate = randint(-10, 10)

    # random translations -
    # trans_1 = randint(-10, 10)
    # trans_2 = randint(-10, 10)
    trans_1 = 0
    trans_2 = 0

    # random zooms
    zoom = uniform(1, 1.3)

    # shearing
    shear_deg = uniform(-25, 25)

    # set the transform parameters for skimage.transform.warp
    # have to shift to center and then shift back after transformation otherwise
    # rotations will make image go out of frame
    center_shift = np.array((size, size)) / 2. - 0.5
    tform_center = transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = transform.SimilarityTransform(translation=center_shift)

    tform_aug = transform.AffineTransform(rotation=np.deg2rad(dorotate),
                                          scale=(1 / zoom, 1 / zoom),
                                          shear=np.deg2rad(shear_deg),
                                          translation=(trans_1, trans_2))

    tform = tform_center + tform_aug + tform_uncenter

    pixels = fast_warp(pixels, tform, output_shape=(size, size))
    return pixels.astype('uint8')


def invert1(pixels):
    amax = np.amax(pixels)
    return np.absolute(pixels - amax)


def invert2(pixels):
    return 255 - pixels


def edges(pixels):
    p = filters.sobel(pixels)
    p = p * 256
    p = p.astype('uint8')
    return p


def combined_random(pixels):
    if randint(0, 10) > 3:
        pixels = warp(pixels)

    my_rand = randint(0, 3)
    if my_rand == 0:
        pixels = invert1(pixels)
    elif my_rand == 1:
        pixels = invert2(pixels)

    if randint(0, 3) == 0:
        pixels = edges(pixels)

    return pixels


def augment(file_path, output_name, *methods):
    image = Image.open(file_path)
    pixels = np.array(image)
    for m in methods:
        pixels = m(pixels)
    result = Image.fromarray(pixels)
    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    result.save('{}/{}.bmp'.format(OUTPUT_PATH, output_name))
    return pixels

# np.set_printoptions(linewidth=1000)
#
# for i in range(10):
#     print("image", i)
#     augment(ORIGINAL_PATH, "aug{}".format(i), combined_random)
