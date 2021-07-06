"""

Library of a collection of encoders and decoders that can readily be imported
and used by the adversarial and convolutional autoencoder models.

Shapes are adjusted to the flow past a cylinder problem.

"""

from keras.layers import Dense, Flatten, Reshape, Conv2D, UpSampling2D, \
                         Cropping2D, MaxPool2D
from keras.models import Sequential

__author__ = "Zef Wolffs"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


def build_custom_conv_encoder(input_shape, latent_dim, initializer,
                              info=False):
    encoder = Sequential()
    encoder.add(Conv2D(32, (5, 5), padding="same", activation="relu",
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(Conv2D(64, (5, 5), strides=(2, 2), activation="relu",
                       padding="same", kernel_initializer=initializer))
    encoder.add(Conv2D(128, (5, 5), strides=(2, 2), activation="relu",
                       padding="same", kernel_initializer=initializer))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    return encoder


def build_custom_conv_decoder(latent_dim, initializer, info=False):
    decoder = Sequential()
    decoder.add(Dense(78848, input_dim=latent_dim,
                      kernel_initializer=initializer))
    decoder.add(Reshape((56, 11, 128)))
    decoder.add(Conv2D(64, (5, 5), activation="relu", padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(32, (5, 5), activation="relu", padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(2, (5, 5), activation="sigmoid", padding="same",
                       kernel_initializer=initializer))
    decoder.add(Cropping2D(cropping=((2, 1), (1, 1))))
    if info:
        print(decoder.summary())

    return decoder


def build_omata_encoder_decoder(input_shape, latent_dim, initializer,
                                info=False, act="elu", dense_act=None):
    """
    This encoder-decoder pair works for 55 by 42 grids
    """
    encoder = Sequential()
    encoder.add(Conv2D(16, (3, 3), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(8, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(8, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(392, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      activation=dense_act))
    decoder.add(Reshape((encoder.layers[6].input_shape[1],
                         encoder.layers[6].input_shape[1], 8)))
    decoder.add(Conv2D(8, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(8, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(16, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(2, (3, 3), activation="linear", padding="same",
                       kernel_initializer=initializer))
    decoder.add(Cropping2D(cropping=((1, 0), (1, 1))))

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_wider_omata_encoder_decoder(input_shape, latent_dim, initializer,
                                      info=False, act="elu", dense_act=None):
    """
    This encoder-decoder pair works for 55 by 42 grids
    """
    encoder = Sequential()
    encoder.add(Conv2D(16, (5, 5), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(32, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(64, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(2688, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      activation=dense_act))
    decoder.add(Reshape((encoder.layers[6].input_shape[1],
                         encoder.layers[6].input_shape[2], 64)))
    decoder.add(Conv2D(64, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(32, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(16, (5, 5), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(2, (3, 3), activation="linear", padding="same",
                       kernel_initializer=initializer))
    decoder.add(Cropping2D(cropping=((1, 0), (3, 3))))

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_wide_omata_encoder_decoder(input_shape, latent_dim, initializer,
                                     info=False, act="elu", dense_act=None):
    """
    relatively wide model omata encoder decoder


    This encoder-decoder pair works for 55 by 42 grids
    """
    encoder = Sequential()
    encoder.add(Conv2D(32, (5, 5), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(64, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(128, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(5376, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      activation=dense_act))
    decoder.add(Reshape((encoder.layers[6].input_shape[1],
                         encoder.layers[6].input_shape[2], 128)))
    decoder.add(Conv2D(128, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(64, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(32, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(2, (3, 3), activation="linear", padding="same",
                kernel_initializer=initializer))
    decoder.add(Cropping2D(cropping=((1, 0), (3, 3))))

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_deeper_omata_encoder_decoder(input_shape, latent_dim, initializer,
                                       info=False, act="elu", dense_act=None):
    """
    This encoder-decoder pair works for 55 by 42 grids
    """
    encoder = Sequential()
    encoder.add(Conv2D(32, (5, 5), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(64, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(64, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(128, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(1536, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      activation=dense_act))
    decoder.add(Reshape((encoder.layers[8].input_shape[1],
                         encoder.layers[8].input_shape[2], 128)))
    decoder.add(Conv2D(128, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(64, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(64, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(32, (3, 3), activation=act, padding="valid",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(2, (3, 3), activation="linear", padding="valid",
                kernel_initializer=initializer))
    decoder.add(Cropping2D(cropping=((2, 1), (0, 0))))

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_denser_omata_encoder_decoder(input_shape, latent_dim, initializer,
                                       info=False, act="elu", dense_act=None):
    """
    This encoder-decoder pair works for 55 by 42 grids
    """
    encoder = Sequential()
    encoder.add(Conv2D(32, (5, 5), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(64, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(128, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(int(5376/2),
                      kernel_initializer=initializer,
                      activation=dense_act))
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(int(5376/2),
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(latent_dim,)))
    decoder.add(Dense(5376,
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(int(5376/2),)))
    decoder.add(Reshape((encoder.layers[6].input_shape[1],
                         encoder.layers[6].input_shape[2], 128)))
    decoder.add(Conv2D(128, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(64, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(32, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(2, (3, 3), activation="linear", padding="same",
                kernel_initializer=initializer))
    decoder.add(Cropping2D(cropping=((1, 0), (3, 3))))

    decoder.build(input_shape)

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_densest_omata_encoder_decoder(input_shape, latent_dim, initializer,
                                        info=False, act="elu", dense_act=None):
    """
    This encoder-decoder pair works for 55 by 42 grids
    """
    encoder = Sequential()
    encoder.add(Conv2D(32, (5, 5), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(64, (3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(int(9856),
                      kernel_initializer=initializer,
                      activation=dense_act))
    encoder.add(Dense(int(9856/2),
                      kernel_initializer=initializer,
                      activation=dense_act))
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(int(9856/2),
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(latent_dim,)))
    decoder.add(Dense(9856,
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(int(9856/2),)))
    decoder.add(Dense(9856,
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(int(9856),)))
    decoder.add(Reshape((encoder.layers[4].input_shape[1],
                         encoder.layers[4].input_shape[2], 64)))
    decoder.add(Conv2D(64, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(32, (3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(2, (3, 3), activation="linear", padding="same",
                kernel_initializer=initializer))
    decoder.add(Cropping2D(cropping=((1, 0), (1, 1))))

    decoder.build(input_shape)

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_agostini_encoder_decoder(input_shape, latent_dim, initializer,
                                   info=False):
    """
    This encoder-decoder pair works for 221 by 42 grids
    """
    encoder = Sequential()
    encoder.add(Conv2D(16, (5, 5), padding="same", activation="relu",
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(32, (3, 3), activation="relu",
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(64, (3, 3), activation="relu",
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(9856, input_dim=latent_dim,
                      kernel_initializer=initializer))
    decoder.add(Reshape((encoder.layers[5].input_shape[1],
                         encoder.layers[5].input_shape[2], 16)))
    decoder.add(Conv2D(64, (3, 3), activation="relu", padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(16, (5, 5), activation="sigmoid", padding="same",
                       kernel_initializer=initializer))
    decoder.add(Conv2D(2, (3, 3), activation="sigmoid", padding="same",
                       kernel_initializer=initializer))
    decoder.add(Cropping2D(cropping=((1, 2), (1, 1))))

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_mnist_wide_omata_encoder_decoder(input_shape, latent_dim,
                                           initializer, info=False):
    """
    This encoder-decoder pair works for 28 by 28 grids
    """
    encoder = Sequential()
    encoder.add(Conv2D(128, (3, 3), padding="same", activation="relu",
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(64, (3, 3), activation="relu",
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Conv2D(32, (3, 3), activation="relu",
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool2D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(784, input_dim=latent_dim,
                      kernel_initializer=initializer))
    decoder.add(Reshape((encoder.layers[5].input_shape[1],
                         encoder.layers[5].input_shape[2], 16)))
    decoder.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(64, (3, 3), activation="relu", padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same",
                       kernel_initializer=initializer))
    decoder.add(Cropping2D(cropping=((0, 0), (0, 0))))

    if info:
        print(decoder.summary())

    return encoder, decoder
