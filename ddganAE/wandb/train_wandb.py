from tensorflow.python.keras.backend import dropout
import wandb
import tensorflow as tf
from ddganAE.models import CAE, AAE, SVDAE, AAE_combined_loss
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
        if config.optimizer == "nadam":
            optimizer = tf.keras.optimizers.Nadam(lr=config.learning_rate, beta_1=config.momentum, beta_2=config.beta_2)
        elif config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate, beta_1=config.momentum, beta_2=config.beta_2)
        elif config.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=config.momentum)

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
        if config.optimizer == "nadam":
            optimizer = tf.keras.optimizers.Nadam(lr=config.learning_rate, beta_1=config.momentum, beta_2=config.beta_2)
        elif config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate, beta_1=config.momentum, beta_2=config.beta_2)
        elif config.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=config.momentum)

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

        if config.train_method == "default":
            aae = AAE(encoder, decoder, discriminator, optimizer)
        elif config.train_method == "combined_loss":
            aae = AAE_combined_loss(encoder, decoder, discriminator, optimizer)
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
        if config.optimizer == "nadam":
            optimizer = tf.keras.optimizers.Nadam(lr=config.learning_rate, beta_1=config.momentum, beta_2=config.beta_2)
        elif config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate, beta_1=config.momentum, beta_2=config.beta_2)
        elif config.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=config.momentum)

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
        elif config.architecture == "vinicius":
            encoder, decoder = build_vinicius_encoder_decoder(100, 10, initializer, act=config.activation,
                                                     dense_act=config.dense_activation, dropout=config.dropout, reg=config.regularization,
                                                     batchnorm=config.batch_normalization)
        elif config.architecture == "smaller_vinicius":
            encoder, decoder = build_smaller_vinicius_encoder_decoder(100, 10, initializer, act=config.activation,
                                                             dense_act=config.dense_activation, dropout=config.dropout, reg=config.regularization,
                                                             batchnorm=config.batch_normalization)
        elif config.architecture == "slimmer_vinicius":
            encoder, decoder = build_slimmer_vinicius_encoder_decoder(100, 10, initializer, act=config.activation,
                                                             dense_act=config.dense_activation, dropout=config.dropout, reg=config.regularization,
                                                             batchnorm=config.batch_normalization)
        svdae = SVDAE(encoder, decoder, optimizer)
        svdae.compile(100, weight_loss=False)
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
        'values': ['denser_omata', 'densest_omata']
      },
      'activation': {
        'values': ['elu']
      },
      'dense_activation': {
        'values': ['relu', None]
      },
      'batch_size': {
        'values': [64, 128]
      },
      'learning_rate': {
        'values': [5e-4, 5e-5, 5e-6]
      },
      'optimizer': {
          'values': ['nadam', 'adam', 'sgd']
      },
      'momentum': {
          'values': [0.8, 0.9, 0.98]
      },
      'beta_2': {
          'values': [0.9, 0.999, 0.99999]
      }
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
        'values': ['denser_omata', 'densest_omata']
      },
      'activation': {
        'values': ['elu']
      },
      'dense_activation': {
        'values': ['relu', None]
      },
      'batch_size': {
        'values': [64, 128]
      },
      'learning_rate': {
        'values': [5e-4, 5e-5, 5e-6]
      },
      'discriminator_architecture': {
        'values': ['custom', 'custom_wider']
      },
      'optimizer': {
          'values': ['nadam', 'adam', 'sgd']
      },
      'momentum': {
          'values': [0.8, 0.9, 0.98]
      },
      'beta_2': {
          'values': [0.9, 0.999, 0.99999]
      }
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
        'values': ['dense', 'deeper_dense', 'wider_dense', 'slimmer_dense', 'vinicius', 'smaller_vinicius', 'slimmer_vinicius']
      },
      'activation': {
        'values': ['relu', 'elu', 'sigmoid']
      },
      'dense_activation': {
        'values': ['relu', 'linear']
      },
      'batch_size': {
        'values': [32, 64, 128]
      },
      'learning_rate': {
        'values': [5e-3, 5e-4, 5e-5]
      },
      'dropout': {
        'values': [0.3, 0.55, 0.8]
      },
      'optimizer': {
          'values': ['nadam', 'adam', 'sgd']
      },
      'momentum': {
          'values': [0.8, 0.9, 0.98]
      },
      'beta_2': {
          'values': [0.9, 0.999, 0.99999]
      },
      'batch_normalization': {
          'values': [True, False]
      },
      'regularization': {
          'values': [1e-4, 1e-5, 0]
      }
    }
}
