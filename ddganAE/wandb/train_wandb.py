import wandb
import tensorflow as tf
from ddganAE.models import CAE, AAE, SVDAE
from ddganAE.architectures import * 
from ddganAE.preprocessing import convert_2d
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split


def train_wandb_cae(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Data processing
        snapshots_grids = np.load(config.datafile)

        input_shape = (55, 42, 2)
        snapshots = convert_2d(snapshots_grids, input_shape, 2000)
        snapshots = np.array(snapshots).reshape(8000, *input_shape)

        # Data normalization
        layer = preprocessing.Normalization()
        layer.adapt(snapshots)

        x_train, x_val = train_test_split(snapshots, test_size=0.1)
        x_train = layer(x_train)
        x_val = layer(x_val)

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        optimizer = tf.keras.optimizers.Nadam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999)

        if config.architecture == "omata":
            encoder, decoder = build_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)
        elif config.architecture == "wider_omata":
            encoder, decoder = build_wider_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)
        elif config.architecture == "deeper_omata":
            encoder, decoder = build_deeper_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)
        elif config.architecture == "wide_omata":
            encoder, decoder = build_wide_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)
        elif config.architecture == "denser_omata":
            encoder, decoder = build_denser_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)
        elif config.architecture == "densest_omata":
            encoder, decoder = build_densest_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)

        cae = CAE(encoder, decoder, optimizer)
        cae.compile(input_shape)

        cae.train(x_train, 200, val_data=x_val, batch_size=config.batch_size, wandb_log=True)


def train_wandb_aae(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Data processing
        snapshots_grids = np.load(config.datafile)

        input_shape = (55, 42, 2)
        snapshots = convert_2d(snapshots_grids, input_shape, 2000)
        snapshots = np.array(snapshots).reshape(8000, *input_shape)

        # Data normalization
        layer = preprocessing.Normalization()
        layer.adapt(snapshots)

        x_train, x_val = train_test_split(snapshots, test_size=0.1)
        x_train = layer(x_train)
        x_val = layer(x_val)

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        optimizer = tf.keras.optimizers.Nadam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999)

        if config.architecture == "omata":
            encoder, decoder = build_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)
        elif config.architecture == "wider_omata":
            encoder, decoder = build_wider_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)
        elif config.architecture == "deeper_omata":
            encoder, decoder = build_deeper_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)
        elif config.architecture == "wide_omata":
            encoder, decoder = build_wide_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)
        elif config.architecture == "denser_omata":
            encoder, decoder = build_denser_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)
        elif config.architecture == "densest_omata":
            encoder, decoder = build_densest_omata_encoder_decoder(input_shape, 10, initializer, info=False, act=config.activation, dense_act=config.dense_activation)

        if config.discriminator_architecture == "custom":
            discriminator = build_custom_discriminator(10, initializer, info=False)
        elif config.discriminator_architecture == "custom_wider":
            discriminator = build_custom_wider_discriminator(10, initializer, info=False)

        aae = AAE(encoder, decoder, discriminator, optimizer)
        aae.compile(input_shape)
        aae.train(x_train, 200, val_data=x_val, batch_size=config.batch_size, wandb_log=True)


def train_wandb_svdae(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Data processing
        snapshots_grids = np.load(config.datafile)

        # Data normalization
        layer = preprocessing.Normalization(axis=None)
        layer.adapt(snapshots_grids)

        snapshots_grids = snapshots_grids.swapaxes(0, 2)

        x_train, x_val = train_test_split(snapshots_grids, test_size=0.1)
        x_train = layer(x_train).numpy().swapaxes(0, 2)
        x_val = layer(x_val).numpy().swapaxes(0, 2)

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        optimizer = tf.keras.optimizers.Nadam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999)

        if config.architecture == "dense":
            encoder = build_dense_encoder(10, initializer, info=False, act=config.activation, dropout=config.dropout)
            decoder = build_dense_decoder(100, 10, initializer, info=False, act=config.activation, dropout=config.dropout)
        elif config.architecture == "deeper_dense":
            encoder = build_deeper_dense_encoder(10, initializer, info=False, act=config.activation, dropout=config.dropout)
            decoder = build_deeper_dense_decoder(100, 10, initializer, info=False, act=config.activation, dropout=config.dropout)
        elif config.architecture == "wider_dense":
            encoder = build_wider_dense_encoder(10, initializer, info=False, act=config.activation, dropout=config.dropout)
            decoder = build_wider_dense_decoder(100, 10, initializer, info=False, act=config.activation, dropout=config.dropout)
        elif config.architecture == "slimmer_dense":
            encoder = build_slimmer_dense_encoder(10, initializer, info=False, act=config.activation, dropout=config.dropout)
            decoder = build_slimmer_dense_decoder(100, 10, initializer, info=False, act=config.activation, dropout=config.dropout)

        svdae = SVDAE(encoder, decoder, optimizer)
        svdae.compile(100)
        svdae.train(x_train, 200, val_data=x_val, batch_size=config.batch_size, wandb_log=True)


# Configuration options for hyperparameter optimization
cae_sweep_config = {
    'method': 'random',
    'metric': {
      'name': 'valid_loss',
      'goal': 'minimize'  
    },
    'parameters': {    
      'architecture': {
        'values': ['omata', 'wider_omata', 'wide_omata', 'denser_omata', 'deeper_omata', 'densest_omata']
      },
      'activation': {
        'values': ['relu', 'elu', 'sigmoid']
      },
      'dense_activation': {
        'values': ['relu', None]
      },
      'batch_size': {
        'values': [32, 64, 128]
      },
      'learning_rate': {
        'values': [5e-1, 5e-2, 5e-3, 5e-4, 5e-5]
      },
    }
}

# Configuration options for hyperparameter optimization
aae_sweep_config = {
    'method': 'random',
    'metric': {
      'name': 'valid_loss',
      'goal': 'minimize'  
    },
    'parameters': {    
      'architecture': {
        'values': ['omata', 'wider_omata', 'wide_omata', 'denser_omata', 'deeper_omata', 'densest_omata']
      },
      'activation': {
        'values': ['relu', 'elu', 'sigmoid']
      },
      'dense_activation': {
        'values': ['relu', None]
      },
      'batch_size': {
        'values': [32, 64, 128]
      },
      'learning_rate': {
        'values': [5e-1, 5e-2, 5e-3, 5e-4, 5e-5]
      },
      'discriminator_architecture': {
        'values': ['custom', 'custom_wider']
      },
    }
}

# Configuration options for hyperparameter optimization
svdae_sweep_config = {
    'method': 'random',
    'metric': {
      'name': 'valid_loss',
      'goal': 'minimize'
    },
    'parameters': {    
      'architecture': {
        'values': ['dense', 'deeper_dense', 'wider_dense', 'slimmer_dense']
      },
      'activation': {
        'values': ['relu', 'elu', 'sigmoid']
      },
      'batch_size': {
        'values': [32, 64, 128]
      },
      'learning_rate': {
        'values': [5e-1, 5e-2, 5e-3, 5e-4, 5e-5]
      },
      'dropout': {
        'values': [0.3, 0.55, 0.8]
      },
    }
}