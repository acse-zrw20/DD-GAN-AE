"""

Library of a collection of encoders and decoders that can readily be imported
and used by the 3D adversarial and convolutional autoencoder models.

"""

from keras.layers import Dense
from keras.models import Sequential

__author__ = "Zef Wolffs"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


def build_custom_discriminator(latent_dim, initializer, info=False):
    discriminator = Sequential()
    discriminator.add(Dense(100, activation='relu',
                            kernel_initializer=initializer,
                            input_dim=latent_dim, bias_initializer=initializer)
                      )
    discriminator.add(Dense(500, activation='relu',
                            kernel_initializer=initializer,
                            bias_initializer=initializer))
    discriminator.add(Dense(1, activation="sigmoid",
                            kernel_initializer=initializer,
                            bias_initializer=initializer))

    if info:
        print(discriminator.summary())

    return discriminator


def build_custom_wider_discriminator(latent_dim, initializer, info=False):
    discriminator = Sequential()
    discriminator.add(Dense(1000, activation='relu',
                            kernel_initializer=initializer,
                            input_dim=latent_dim, bias_initializer=initializer)
                      )
    discriminator.add(Dense(1000, activation='relu',
                            kernel_initializer=initializer,
                            bias_initializer=initializer))
    discriminator.add(Dense(1, activation="sigmoid",
                            kernel_initializer=initializer,
                            bias_initializer=initializer))

    if info:
        print(discriminator.summary())

    return discriminator
