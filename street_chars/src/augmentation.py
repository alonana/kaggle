from PIL import Image


def transform(image_pixels):
    return image_pixels


def augment(file_path):
    image = Image.open(file_path)
    image_pixels = [p for p in list(image.getdata())]
    print(image_pixels)
    transform(image_pixels)
    Image.
    return image_pixels


augment('/home/alon/git/kaggle/street_chars/output/gray_scale/trainResized/1.Bmp')
