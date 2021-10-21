from keras.models import *
from keras.layers import *
import keras.backend as K

import tensorflow as tf
from .config import IMAGE_ORDERING


def vanilla_encoder(input_height=224,  input_width=224, channels=3):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    if IMAGE_ORDERING == 'channels_first':
        img_input = tf.keras.Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = tf.keras.Input(shape=(input_height, input_width, channels))

    x = img_input
    levels = []

    x = (tf.keras.layers.ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (tf.keras.layers.Conv2D(filter_size, (kernel, kernel),
                data_format=IMAGE_ORDERING, padding='valid'))(x)
    x = (tf.keras.layers.BatchNormalization())(x)
    x = (tf.keras.layers.Activation('relu'))(x)
    x = (tf.keras.layers.MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    x = (tf.keras.layers.ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (tf.keras.layers.Conv2D(128, (kernel, kernel), data_format=IMAGE_ORDERING,
         padding='valid'))(x)
    x = (tf.keras.layers.BatchNormalization())(x)
    x = (tf.keras.layers.Activation('relu'))(x)
    x = (tf.keras.layers.MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    for _ in range(3):
        x = (tf.keras.layers.ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
        x = (tf.keras.layers.Conv2D(256, (kernel, kernel),
                    data_format=IMAGE_ORDERING, padding='valid'))(x)
        x = (tf.keras.layers.BatchNormalization())(x)
        x = (tf.keras.layers.Activation('relu'))(x)
        x = (tf.keras.layers.MaxPooling2D((pool_size, pool_size),
             data_format=IMAGE_ORDERING))(x)
        levels.append(x)

    return img_input, levels
