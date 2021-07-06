"""

Library of a collection of encoders and decoders that can readily be imported
and used by the 3D adversarial and convolutional autoencoder models.

Shapes are adjusted to the slug flow problem.

"""

from keras.layers import Dense, Flatten, Reshape, Conv3D, UpSampling3D, \
                         Cropping3D, MaxPool3D
from keras.models import Sequential

__author__ = "Zef Wolffs"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


def build_omata_encoder_decoder(input_shape, latent_dim, initializer,
                                info=False, act="elu", dense_act=None):
    """
    This encoder-decoder pair works for 55 by 42 grids
    """
    encoder = Sequential()
    encoder.add(Conv3D(16, (3, 3, 3), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(8, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(8, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(576, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      activation=dense_act))
    decoder.add(Reshape((encoder.layers[6].input_shape[1],
                         encoder.layers[6].input_shape[2],
                         encoder.layers[6].input_shape[3], 8)))
    decoder.add(Conv3D(8, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(8, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(16, (3, 3, 3), activation=act, padding="valid",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(4, (3, 3, 3), activation="linear", padding="same",
                       kernel_initializer=initializer))
    decoder.add(Cropping3D(cropping=((0, 0), (0, 0), (0, 0))))

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_denser_omata_encoder_decoder(input_shape, latent_dim, initializer,
                                       info=False, act="elu", dense_act=None):
    """

    """
    encoder = Sequential()
    encoder.add(Conv3D(32, (3, 3, 3), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(64, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(128, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(int(9216/2),
                      kernel_initializer=initializer,
                      activation=dense_act))
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(int(9216/2),
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(latent_dim,)))
    decoder.add(Dense(9216,
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(int(9216/2),)))
    decoder.add(Reshape((encoder.layers[6].input_shape[1],
                         encoder.layers[6].input_shape[2],
                         encoder.layers[6].input_shape[3], 128)))
    decoder.add(Conv3D(128, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(64, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(32, (3, 3, 3), activation=act, padding="valid",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(4, (3, 3, 3), activation="linear", padding="same",
                kernel_initializer=initializer))
    decoder.add(Cropping3D(cropping=((0, 0), (0, 0), (0, 0))))

    decoder.build(input_shape)

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_densest_omata_encoder_decoder(input_shape, latent_dim,
                                        initializer, info=False,
                                        act="elu", dense_act=None):
    """

    """
    encoder = Sequential()
    encoder.add(Conv3D(32, (3, 3, 3), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(64, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(int(24000),
                      kernel_initializer=initializer,
                      activation=dense_act))
    encoder.add(Dense(int(24000/2),
                      kernel_initializer=initializer,
                      activation=dense_act))
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(int(24000/2),
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(latent_dim,)))
    decoder.add(Dense(24000,
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(int(24000/2),)))
    decoder.add(Dense(24000,
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(int(24000),)))
    decoder.add(Reshape((encoder.layers[4].input_shape[1],
                         encoder.layers[4].input_shape[2],
                         encoder.layers[4].input_shape[3], 64)))
    decoder.add(Conv3D(64, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(32, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(4, (3, 3, 3), activation="linear", padding="same",
                kernel_initializer=initializer))
    decoder.add(Cropping3D(cropping=((0, 0), (0, 0), (0, 0))))

    decoder.build(input_shape)

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_densest_thinner_omata_encoder_decoder(input_shape, latent_dim,
                                                initializer,
                                                info=False, act="elu",
                                                dense_act=None):
    """

    """
    encoder = Sequential()
    encoder.add(Conv3D(16, (3, 3, 3), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(32, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(int(12000),
                      kernel_initializer=initializer,
                      activation=dense_act))
    encoder.add(Dense(int(12000/2),
                      kernel_initializer=initializer,
                      activation=dense_act))
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(int(12000/2),
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(latent_dim,)))
    decoder.add(Dense(12000,
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(int(12000/2),)))
    decoder.add(Dense(12000,
                      kernel_initializer=initializer,
                      activation=dense_act,
                      input_shape=(int(12000),)))
    decoder.add(Reshape((encoder.layers[4].input_shape[1],
                         encoder.layers[4].input_shape[2],
                         encoder.layers[4].input_shape[3], 32)))
    decoder.add(Conv3D(32, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(16, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(4, (3, 3, 3), activation="linear", padding="same",
                kernel_initializer=initializer))
    decoder.add(Cropping3D(cropping=((0, 0), (0, 0), (0, 0))))

    decoder.build(input_shape)

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_wide_omata_encoder_decoder(input_shape, latent_dim, initializer,
                                     info=False, act="elu", dense_act=None):
    """
    relatively wide model omata encoder decoder

    """
    encoder = Sequential()
    encoder.add(Conv3D(32, (3, 3, 3), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(64, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(128, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(9216, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      activation=dense_act))
    decoder.add(Reshape((encoder.layers[6].input_shape[1],
                         encoder.layers[6].input_shape[2],
                         encoder.layers[6].input_shape[3], 128)))
    decoder.add(Conv3D(128, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(64, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(32, (3, 3, 3), activation=act, padding="valid",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(4, (3, 3, 3), activation="linear", padding="same",
                kernel_initializer=initializer))
    decoder.add(Cropping3D(cropping=((0, 0), (0, 0), (0, 0))))

    if info:
        print(decoder.summary())

    return encoder, decoder


def build_deeper_omata_encoder_decoder(input_shape, latent_dim, initializer,
                                       info=False, act="elu", dense_act=None):
    """
    This encoder-decoder pair works for 55 by 42 grids
    """
    encoder = Sequential()
    encoder.add(Conv3D(32, (5, 5, 5), padding="same", activation=act,
                       input_shape=input_shape,
                       kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(64, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(64, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Conv3D(128, (3, 3, 3), activation=act,
                       padding="same", kernel_initializer=initializer))
    encoder.add(MaxPool3D(padding="same"))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim, activation="linear"))

    if info:
        print(encoder.summary())

    decoder = Sequential()
    decoder.add(Dense(2048, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      activation=dense_act))
    decoder.add(Reshape((encoder.layers[8].input_shape[1],
                         encoder.layers[8].input_shape[2],
                         encoder.layers[8].input_shape[3], 128)))
    decoder.add(Conv3D(128, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(64, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(64, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(32, (3, 3, 3), activation=act,
                       padding="valid",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(4, (3, 3, 3), activation="linear",
                       padding="same",
                kernel_initializer=initializer))
    decoder.add(Cropping3D(cropping=((0, 0), (4, 4), (4, 4))))

    if info:
        print(decoder.summary())

    return encoder, decoder
