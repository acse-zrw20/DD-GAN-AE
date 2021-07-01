"""

Library of a collection of encoders and decoders that can readily be imported
and used by the 3D adversarial and convolutional autoencoder models.

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
                         encoder.layers[6].input_shape[1],
                         encoder.layers[6].input_shape[3], 8)))
    decoder.add(Conv3D(8, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(8, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(16, (3, 3, 3), activation=act, padding="same",
                       kernel_initializer=initializer))
    decoder.add(UpSampling3D())
    decoder.add(Conv3D(4, (3, 3, 3), activation="linear", padding="same",
                       kernel_initializer=initializer))
    decoder.add(Cropping3D(cropping=((2, 2), (2, 2), (2, 2))))

    if info:
        print(decoder.summary())

    return encoder, decoder
