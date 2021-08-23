"""

Functions used for weights and biases hyperparameter optimization of
autoencoders on slug flow dataset

"""

import wandb
import tensorflow as tf
import argparse
import os
import json
from sklearn.preprocessing import MinMaxScaler
from ddganAE.models import CAE, AAE, SVDAE, AAE_combined_loss
from ddganAE.architectures.cae.D3 import (
    build_omata_encoder_decoder,
    build_deeper_omata_encoder_decoder,
    build_wide_omata_encoder_decoder,
    build_denser_omata_encoder_decoder,
    build_densest_omata_encoder_decoder,
    build_densest_thinner_omata_encoder_decoder
)
from ddganAE.architectures.svdae import (
    build_vinicius_encoder_decoder,
    build_slimmer_vinicius_encoder_decoder,
    build_smaller_vinicius_encoder_decoder,
    build_dense_decoder,
    build_deeper_dense_encoder,
    build_dense_encoder,
    build_slimmer_dense_decoder,
    build_wider_dense_decoder,
    build_wider_dense_encoder,
    build_deeper_dense_decoder,
    build_slimmer_dense_encoder,
)
from ddganAE.architectures.discriminators import (
    build_custom_discriminator,
    build_custom_wider_discriminator
)
import numpy as np
from sklearn.model_selection import train_test_split

__author__ = "Zef Wolffs"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


def train_wandb_cae(config=None):
    """
    Construct and subsequently train the model while reporting losses to
    weights and biases platform. Weights and biases also controls
    hyperparameters.

    Args:
        config (dict, optional): Dictionary with hyperparameters, set by
                                 weights and biases. Defaults to None.
    """
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Data processing
        snapshots_grids = np.load(config.datafile)

        # Set all >0 to 0 and all <1 to 1 for alpha field
        snapshots_grids[:, :, :, :, 3][
            np.where(snapshots_grids[:, :, :, :, 3] < 0)] = 0
        snapshots_grids[:, :, :, :, 3][
            np.where(snapshots_grids[:, :, :, :, 3] > 1)] = 1

        # Rescale all the velocity fields
        scaler = MinMaxScaler()
        snapshots_grids[:, :, :, :, :3] = \
            scaler.fit_transform(
                snapshots_grids[:, :, :, :, :3].reshape(-1, 1)).reshape(
                    snapshots_grids[:, :, :, :, :3].shape)

        x_train, x_val = train_test_split(snapshots_grids, test_size=0.2)

        input_shape = (60, 20, 20, 4)

        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=None
        )
        if config.optimizer == "nadam":
            optimizer = tf.keras.optimizers.Nadam(
                lr=config.learning_rate,
                beta_1=config.momentum,
                beta_2=config.beta_2,
            )
        elif config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(
                lr=config.learning_rate,
                beta_1=config.momentum,
                beta_2=config.beta_2,
            )
        elif config.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=config.learning_rate, momentum=config.momentum
            )

        if config.architecture == "omata":
            encoder, decoder = build_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )
        elif config.architecture == "densest_thinner_omata":
            encoder, decoder = build_densest_thinner_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )
        elif config.architecture == "deeper_omata":
            encoder, decoder = build_deeper_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )
        elif config.architecture == "wide_omata":
            encoder, decoder = build_wide_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )
        elif config.architecture == "denser_omata":
            encoder, decoder = build_denser_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )
        elif config.architecture == "densest_omata":
            encoder, decoder = build_densest_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )

        cae = CAE(encoder, decoder, optimizer)
        cae.compile(input_shape)

        cae.train(
            x_train,
            100,
            val_data=x_val,
            batch_size=config.batch_size,
            wandb_log=True,
        )

        if config.savemodel:
            dirname = "model_" + wandb.run.name
            os.mkdir(dirname)
            cae.encoder.save(dirname + '/encoder')
            cae.decoder.save(dirname + '/decoder')


def train_wandb_aae(config=None):
    """
    Construct and subsequently train the model while reporting losses to
    weights and biases platform. Weights and biases also controls
    hyperparameters.

    Args:
        config (dict, optional): Dictionary with hyperparameters, set by
                                 weights and biases. Defaults to None.
    """
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Data processing
        snapshots_grids = np.load(config.datafile)

        # Set all >0 to 0 and all <1 to 1 for alpha field
        snapshots_grids[:, :, :, :, 3][
            np.where(snapshots_grids[:, :, :, :, 3] < 0)] = 0
        snapshots_grids[:, :, :, :, 3][
            np.where(snapshots_grids[:, :, :, :, 3] > 1)] = 1

        # Rescale all the velocity fields
        scaler = MinMaxScaler()
        snapshots_grids[:, :, :, :, :3] = \
            scaler.fit_transform(
                snapshots_grids[:, :, :, :, :3].reshape(-1, 1)).reshape(
                    snapshots_grids[:, :, :, :, :3].shape)

        x_train, x_val = train_test_split(snapshots_grids, test_size=0.2)

        input_shape = (60, 20, 20, 4)

        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=None
        )
        if config.optimizer == "nadam":
            optimizer = tf.keras.optimizers.Nadam(
                lr=config.learning_rate,
                beta_1=config.momentum,
                beta_2=config.beta_2,
            )
        elif config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(
                lr=config.learning_rate,
                beta_1=config.momentum,
                beta_2=config.beta_2,
            )
        elif config.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=config.learning_rate, momentum=config.momentum
            )

        if config.architecture == "omata":
            encoder, decoder = build_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )
        elif config.architecture == "densest_thinner_omata":
            encoder, decoder = build_densest_thinner_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )
        elif config.architecture == "deeper_omata":
            encoder, decoder = build_deeper_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )
        elif config.architecture == "wide_omata":
            encoder, decoder = build_wide_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )
        elif config.architecture == "denser_omata":
            encoder, decoder = build_denser_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )
        elif config.architecture == "densest_omata":
            encoder, decoder = build_densest_omata_encoder_decoder(
                input_shape,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dense_act=config.dense_activation,
                final_act=config.final_act
            )

        if config.discriminator_architecture == "custom":
            discriminator = build_custom_discriminator(
                config.latent_vars, initializer, info=False
            )
        elif config.discriminator_architecture == "custom_wider":
            discriminator = build_custom_wider_discriminator(
                config.latent_vars, initializer, info=False
            )

        if config.train_method == "default":
            aae = AAE(encoder, decoder, discriminator, optimizer)
        elif config.train_method == "combined_loss":
            aae = AAE_combined_loss(encoder, decoder, discriminator, optimizer)
        aae.compile(input_shape)
        aae.train(
            x_train,
            config.epochs,
            val_data=x_val,
            batch_size=config.batch_size,
            wandb_log=True,
        )

        if config.savemodel:
            dirname = "model_" + wandb.run.name
            os.mkdir(dirname)
            aae.encoder.save(dirname + '/encoder')
            aae.decoder.save(dirname + '/decoder')
            aae.discriminator.save(dirname + '/discriminator')


def train_wandb_svdae(config=None):
    """
    Construct and subsequently train the model while reporting losses to
    weights and biases platform. Weights and biases also controls
    hyperparameters.

    Args:
        config (dict, optional): Dictionary with hyperparameters, set by
                                 weights and biases. Defaults to None.
    """
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Data processing
        snapshots_grids = np.load(config.datafile)

        # Set all >0 to 0 and all <1 to 1 for alpha field
        snapshots_grids[:, :, :, :, 3][
            np.where(snapshots_grids[:, :, :, :, 3] < 0)] = 0
        snapshots_grids[:, :, :, :, 3][
            np.where(snapshots_grids[:, :, :, :, 3] > 1)] = 1

        # Rescale all the velocity fields
        scaler = MinMaxScaler()
        snapshots_grids[:, :, :, :, :3] = \
            scaler.fit_transform(
                snapshots_grids[:, :, :, :, :3].reshape(-1, 1)).reshape(
                    snapshots_grids[:, :, :, :, :3].shape)

        x_train, x_val = train_test_split(snapshots_grids, test_size=0.2)

        # Do a little data reshaping for POD
        x_train_long = np.moveaxis(np.moveaxis(x_train, 4, 0), 1, 4).reshape(
            (x_train.shape[-1],
             x_train.shape[1]*x_train.shape[2]*x_train.shape[3],
             x_train.shape[0]))
        x_val_long = np.moveaxis(np.moveaxis(x_val, 4, 0), 1, 4).reshape(
            (x_val.shape[-1], x_val.shape[1]*x_val.shape[2]*x_val.shape[3],
             x_val.shape[0]))

        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=None
        )
        if config.optimizer == "nadam":
            optimizer = tf.keras.optimizers.Nadam(
                lr=config.learning_rate,
                beta_1=config.momentum,
                beta_2=config.beta_2,
            )
        elif config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(
                lr=config.learning_rate,
                beta_1=config.momentum,
                beta_2=config.beta_2,
            )
        elif config.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=config.learning_rate, momentum=config.momentum
            )

        if config.architecture == "dense":
            encoder = build_dense_encoder(
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
            )
            decoder = build_dense_decoder(
                100,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
            )
        elif config.architecture == "deeper_dense":
            encoder = build_deeper_dense_encoder(
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
            )
            decoder = build_deeper_dense_decoder(
                100,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
            )
        elif config.architecture == "wider_dense":
            encoder = build_wider_dense_encoder(
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
            )
            decoder = build_wider_dense_decoder(
                100,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
            )
        elif config.architecture == "slimmer_dense":
            encoder = build_slimmer_dense_encoder(
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
            )
            decoder = build_slimmer_dense_decoder(
                100,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
            )
        elif config.architecture == "vinicius":
            encoder, decoder = build_vinicius_encoder_decoder(
                100,
                config.latent_vars,
                initializer,
                act=config.activation,
                dense_act=config.dense_activation,
                dropout=config.dropout,
                reg=config.regularization,
                batchnorm=config.batch_normalization,
            )
        elif config.architecture == "smaller_vinicius":
            encoder, decoder = build_smaller_vinicius_encoder_decoder(
                100,
                config.latent_vars,
                initializer,
                act=config.activation,
                dense_act=config.dense_activation,
                dropout=config.dropout,
                reg=config.regularization,
                batchnorm=config.batch_normalization,
            )
        elif config.architecture == "slimmer_vinicius":
            encoder, decoder = build_slimmer_vinicius_encoder_decoder(
                100,
                config.latent_vars,
                initializer,
                act=config.activation,
                dense_act=config.dense_activation,
                dropout=config.dropout,
                reg=config.regularization,
                batchnorm=config.batch_normalization,
            )
        svdae = SVDAE(encoder, decoder, optimizer)
        svdae.compile(100, weight_loss=False)
        svdae.train(
            x_train_long,
            100,
            val_data=x_val_long,
            batch_size=config.batch_size,
            wandb_log=True,
        )

        if config.savemodel:
            dirname = "model_" + wandb.run.name
            os.mkdir(dirname)
            svdae.encoder.save(dirname + '/encoder')
            svdae.decoder.save(dirname + '/decoder')
            np.save(dirname + "/R.npy", svdae.R)


# Configuration options for hyperparameter optimization
cae_sweep_config = {
    "method": "random",
    "metric": {"name": "valid_loss", "goal": "minimize"},
    "parameters": {
        "architecture": {"values": ["denser_omata",
                                    "densest_thinner_omata", "omata",
                                    "wide_omata"]},
        "activation": {"values": ["elu", "sigmoid", "relu", "tanh"]},
        "dense_activation": {"values": ["relu", None]},
        "batch_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [5e-3, 5e-4, 5e-5, 5e-6]},
        "optimizer": {"values": ["nadam", "adam", "sgd"]},
        "momentum": {"values": [0.8, 0.9, 0.98]},
        "beta_2": {"values": [0.9, 0.999, 0.99999]},
        "savemodel": {"values": [False]},
        "latent_vars": {"values": [10, 100]},
        "final_act": {
            "values": [
              "linear",
              "sigmoid"
            ]
        }
    },
}

# Configuration options for hyperparameter optimization
aae_sweep_config = {
    "method": "bayes",
    "metric": {"name": "g_valid_loss", "goal": "minimize"},
    "parameters": {
        "architecture": {"values": ["denser_omata",
                                    "densest_thinner_omata", "omata",
                                    "wide_omata"]},
        "activation": {"values": ["elu", "sigmoid", "relu", "tanh"]},
        "dense_activation": {"values": ["relu", None]},
        "batch_size": {"values": [32, 64, 128]},
        "train_method": {"values": ["combined_loss"]},
        "learning_rate": {"values": [5e-3, 5e-4, 5e-5, 5e-6]},
        "discriminator_architecture": {"values": ["custom", "custom_wider"]},
        "optimizer": {"values": ["nadam", "adam", "sgd"]},
        "momentum": {"values": [0.8, 0.9, 0.98]},
        "beta_2": {"values": [0.9, 0.999, 0.99999]},
        "savemodel": {"values": [False]},
        "latent_vars": {"values": [10, 20, 100]},
        "final_act": {
            "values": [
              "linear",
              "sigmoid"
            ]
        },
        "epochs": {
            "values":
            [100, 200, 500, 1000]
        }
    },
}

# Configuration options for hyperparameter optimization
svdae_sweep_config = {
    "method": "random",
    "metric": {"name": "valid_loss", "goal": "minimize"},
    "parameters": {
        "architecture": {
            "values": [
                "dense",
                "deeper_dense",
                "wider_dense",
                "slimmer_dense",
                "vinicius",
                "smaller_vinicius",
                "slimmer_vinicius",
            ]
        },
        "activation": {"values": ["relu", "elu", "sigmoid", "tanh"]},
        "dense_activation": {"values": ["relu", "linear"]},
        "batch_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [5e-3, 5e-4, 5e-5]},
        "dropout": {"values": [0.3, 0.55, 0.8]},
        "optimizer": {"values": ["nadam", "adam", "sgd"]},
        "momentum": {"values": [0.8, 0.9, 0.98]},
        "beta_2": {"values": [0.9, 0.999, 0.99999]},
        "batch_normalization": {"values": [True, False]},
        "regularization": {"values": [1e-4, 1e-5, 0]},
        "savemodel": {"values": [False]},
        "latent_vars": {"values": [10]},
    },
}

# Build a small CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do hyperparameter \
optimization on slug flow dataset")
    parser.add_argument('--model', type=str, nargs='?',
                        default="cae",
                        help='select the model to run hyperparameter \
optimization with')
    parser.add_argument('--datafile', type=str, nargs='?',
                        default="processed/sf_snapshots_200timesteps_rand.npy",
                        help='path to structured grid data file')
    parser.add_argument('--savemodel', type=str, nargs='?',
                        default="False",
                        help='Wether or not to save the models, set "True" for \
saving')
    parser.add_argument('--niters', type=int, nargs='?',
                        default=200,
                        help='Number of sweeps to execute')
    parser.add_argument('--custom_config', type=str, nargs='?',
                        default=None,
                        help='json file with custom configurations for sweep')
    args = parser.parse_args()

    arg_dict = vars(args)

    if arg_dict['model'] == "cae":
        if arg_dict['custom_config'] is not None:
            with open(arg_dict["custom_config"]) as json_file:
                cae_sweep_config = json.load(json_file)

        if arg_dict["savemodel"] == "True":
            cae_sweep_config['parameters']['savemodel'] = \
                {'values': [True]}

        cae_sweep_config['parameters']['datafile'] = \
            {'values': [arg_dict['datafile']]}

        sweep_id = wandb.sweep(cae_sweep_config, project='cae-sf',
                               entity='zeff020')
        wandb.agent(sweep_id, train_wandb_cae, count=arg_dict['niters'])

    if arg_dict['model'] == "aae":
        if arg_dict['custom_config'] is not None:
            with open(arg_dict["custom_config"]) as json_file:
                aae_sweep_config = json.load(json_file)

        if arg_dict["savemodel"] == "True":
            aae_sweep_config['parameters']['savemodel'] = \
                {'values': [True]}

        aae_sweep_config['parameters']['datafile'] = \
            {'values': [arg_dict['datafile']]}

        sweep_id = wandb.sweep(aae_sweep_config, project='aae-sf',
                               entity='zeff020')
        wandb.agent(sweep_id, train_wandb_aae, count=arg_dict['niters'])

    if arg_dict['model'] == "svdae":
        if arg_dict['custom_config'] is not None:
            with open(arg_dict["custom_config"]) as json_file:
                svdae_sweep_config = json.load(json_file)

        if arg_dict["savemodel"] == "True":
            svdae_sweep_config['parameters']['savemodel'] = \
                {'values': [True]}

        svdae_sweep_config['parameters']['datafile'] = \
            {'values': [arg_dict['datafile']]}

        sweep_id = wandb.sweep(svdae_sweep_config, project='svdae-sf',
                               entity='zeff020')
        wandb.agent(sweep_id, train_wandb_svdae, count=arg_dict['niters'])
