from keras.layers import Dense
from keras.models import Sequential


# We make the encoder model
def build_dense_encoder(latent_dim, initializer, info=False):
    encoder = Sequential()
    encoder.add(Dense(1000, activation='relu', kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(1000, activation='relu', kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(latent_dim, kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary())

    return encoder


# We make the decoder model
def build_dense_decoder(input_dim, latent_dim, initializer, info=False):
    decoder = Sequential()
    decoder.add(Dense(1000, activation='relu', input_dim=latent_dim,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(1000, activation='relu', kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(input_dim, activation='sigmoid',
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(decoder.summary())

    return decoder
