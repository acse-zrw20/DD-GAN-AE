from keras.layers import Dense
from keras.models import Sequential


# We make the encoder model
def build_dense_encoder(latent_dim, initializer, info=False,
                        act='relu'):
    encoder = Sequential()
    encoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(latent_dim, kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary())

    return encoder


# We make the decoder model
def build_dense_decoder(input_dim, latent_dim, initializer, info=False,
                        act='relu'):
    decoder = Sequential()
    decoder.add(Dense(1000, activation=act, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(input_dim, activation='sigmoid',
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(decoder.summary())

    return decoder


# We make the encoder model
def build_wider_dense_encoder(latent_dim, initializer, info=False,
                              act='relu'):
    encoder = Sequential()
    encoder.add(Dense(1500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(2000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(latent_dim, kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary())

    return encoder


# We make the decoder model
def build_wider_dense_decoder(input_dim, latent_dim, initializer, info=False,
                              act='relu'):
    decoder = Sequential()
    decoder.add(Dense(1500, activation=act, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(2000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(input_dim, activation='sigmoid',
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(decoder.summary())

    return decoder

# We make the encoder model
def build_slimmer_dense_encoder(latent_dim, initializer, info=False,
                              act='relu'):
    encoder = Sequential()
    encoder.add(Dense(500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(latent_dim, kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary())

    return encoder


# We make the decoder model
def build_slimmer_dense_decoder(input_dim, latent_dim, initializer, info=False,
                              act='relu'):
    decoder = Sequential()
    decoder.add(Dense(500, activation=act, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(input_dim, activation='sigmoid',
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(decoder.summary())

    return decoder


# We make the encoder model
def build_deeper_dense_encoder(latent_dim, initializer, info=False,
                               act='relu'):
    encoder = Sequential()
    encoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    encoder.add(Dense(latent_dim, kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(encoder.summary())

    return encoder


# We make the decoder model
def build_deeper_dense_decoder(input_dim, latent_dim, initializer, info=False,
                               act='relu'):
    decoder = Sequential()
    decoder.add(Dense(1000, activation=act, input_dim=latent_dim,
                      kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(1000, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(500, activation=act, kernel_initializer=initializer,
                      bias_initializer=initializer))
    decoder.add(Dense(input_dim, activation='sigmoid',
                      kernel_initializer=initializer,
                      bias_initializer=initializer))

    if info:
        print(decoder.summary())

    return decoder