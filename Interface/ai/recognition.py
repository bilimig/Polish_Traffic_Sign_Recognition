import os

import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from skimage import transform
from flask import current_app as app
from helpers import constants


def testing(filename):
    image = Image.open(filename)
    image = image.resize((32, 32))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 32, 32, 3)
    return image_arr

#
# def load(filename):
#     np_image = Image.open(filename)
#     np_image = np.array(np_image).astype('float32') / 255
#     np_image = transform.resize(np_image, (32, 32, 3))
#     #np_image = np.reshape(np_image, (1, 32, 32, 3))
#     np_image = np.expand_dims(np_image, axis=0)
#     return np_image
#
#
# def process_image(img_path):
#     model = tf.keras.models.load_model(app.config['MODEL_PATH'])
#     input_data = load(img_path)
#     predictions = model.predict(input_data)
#     label_found = np.argmax(predictions, axis=1)
#     img_reshaped = np.squeeze(input_data, axis=0)
#     img_resized = Image.fromarray(img_reshaped)
#     impath = os.path.join(app.config['RESIZED_FOLDER'], 'test.jpg')
#     img_resized.save(impath)
#     if label_found[0] in constants.LABELS.keys():
#         return constants.LABELS[label_found[0] + 1], predictions[0][label_found][0], impath
#     return "", 0, ""

def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image.convert('RGB')) #.astype('float32') % 256
    np_image = transform.resize(np_image, (32, 32, 3))
    if np_image.ndim == 2 or np_image.shape[2] == 1:
        np_image = np.stack((np_image,) * 3, axis=-1)
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def process_image(img_path):
    model = tf.keras.models.load_model(app.config['MODEL_PATH'])
    input_data = load(img_path)
    predictions = model.predict(input_data)
    label_found = np.argmax(predictions, axis=1)

    img_reshaped = np.squeeze(input_data, axis=0)
    img_reshaped = (img_reshaped * 255).astype(np.uint8)  # Scale back and convert to uint8

    img_resized = Image.fromarray(img_reshaped)
    impath = os.path.join(app.config['RESIZED_FOLDER'], 'test.jpg')
    img_resized.save(impath)

    if label_found[0] in constants.LABELS.keys():
        return constants.LABELS[label_found[0] + 1], predictions[0][label_found][0], impath
    return "", 0, ""

