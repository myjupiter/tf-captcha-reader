import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os
import pdb
from scipy.misc import imresize

def preprocess(img):
    width, height = img.shape[0], img.shape[1]
    img = image.array_to_img(img, scale=False)

    # Crop 48x48px
    desired_width, desired_height = 48, 48

    if width < desired_width:
        desired_width = width
    start_x = np.maximum(0, int((width-desired_width)/2))

    img = img.crop((start_x, np.maximum(0, height-desired_height), start_x+desired_width, height))
    img = img.resize((48, 48))

    img = image.img_to_array(img)
    return img / 255.

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    preprocessing_function=preprocess)

generator = datagen.flow_from_directory(
    'numbers_train', 
    target_size=(48,48),
    batch_size=1024, # Only 405 images in directory, so batch always the same
    classes=['02'],
    shuffle=False,
    class_mode='sparse')

inputs, targets = next(generator)

folder = 'numbers_train/02'
files = os.listdir(folder)
files = list(map(lambda x: os.path.join(folder, x), files))

images = []
for f in files:
    img = image.load_img(f)
    #img = img.resize((48, 48))
    img = image.img_to_array(img)
    img = preprocess(img)

    images.append(img)
inputs2 = np.asarray(images)

print(np.mean(inputs))
print(np.mean(inputs2))