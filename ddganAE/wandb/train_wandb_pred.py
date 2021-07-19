"""

Functions used for weights and biases hyperparameter optimization on slug flow
dataset

"""

import wandb
import tensorflow as tf
import argparse
import os
import json
from sklearn.preprocessing import MinMaxScaler
from ddganAE.models import Predictive_adversarial
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


def train_wandb_pred_aae(config=None):
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
        latent_vars = np.load(config.datafile)

        latent_vars_reshaped = np.moveaxis(latent_vars.reshape(800, 10, 10),
                                           0, 2)

        train_data = latent_vars_reshaped[:4]

        # Scaling the latent variables
        scaler = MinMaxScaler((-1, 1))
        train_data = scaler.fit_transform(
            train_data.reshape(-1, 1)).reshape(train_data.shape)

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
                config.in_vars,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
                final_act=config.final_act
            )
        elif config.architecture == "deeper_dense":
            encoder = build_deeper_dense_encoder(
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout
            )
            decoder = build_deeper_dense_decoder(
                config.in_vars,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
                final_act=config.final_act
            )
        elif config.architecture == "wider_dense":
            encoder = build_wider_dense_encoder(
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout
            )
            decoder = build_wider_dense_decoder(
                config.in_vars,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
                final_act=config.final_act
            )
        elif config.architecture == "slimmer_dense":
            encoder = build_slimmer_dense_encoder(
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout
            )
            decoder = build_slimmer_dense_decoder(
                config.in_vars,
                config.latent_vars,
                initializer,
                info=False,
                act=config.activation,
                dropout=config.dropout,
                final_act=config.final_act
            )
        elif config.architecture == "vinicius":
            encoder, decoder = build_vinicius_encoder_decoder(
                config.in_vars,
                config.latent_vars,
                initializer,
                act=config.activation,
                dense_act=config.dense_activation,
                dropout=config.dropout,
                reg=config.regularization,
                batchnorm=config.batch_normalization,
                final_act=config.final_act
            )
        elif config.architecture == "smaller_vinicius":
            encoder, decoder = build_smaller_vinicius_encoder_decoder(
                config.in_vars,
                config.latent_vars,
                initializer,
                act=config.activation,
                dense_act=config.dense_activation,
                dropout=config.dropout,
                reg=config.regularization,
                batchnorm=config.batch_normalization,
                final_act=config.final_act
            )
        elif config.architecture == "slimmer_vinicius":
            encoder, decoder = build_slimmer_vinicius_encoder_decoder(
                config.in_vars,
                config.latent_vars,
                initializer,
                act=config.activation,
                dense_act=config.dense_activation,
                dropout=config.dropout,
                reg=config.regularization,
                batchnorm=config.batch_normalization,
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

        pred_adv = Predictive_adversarial(encoder, decoder, discriminator, 
                                          optimizer)
        pred_adv.compile(config.in_vars)
        pred_adv.train(
            train_data,
            1000,
            interval=config.interval,
            batch_size=config.batch_size,
            val_size=0.1,
            wandb_log=True,
            noise_std=config.noise_std
        )

        if config.savemodel:
            dirname = "model_" + wandb.run.name
            os.mkdir(dirname)
            pred_adv.encoder.save(dirname + '/encoder')
            pred_adv.decoder.save(dirname + '/decoder')


# Configuration options for hyperparameter optimization
Predictive_adversarial_sweep_config = {
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
        "discriminator_architecture": {"values": ["custom", "custom_wider"]},
        "in_vars": {"values": [10]},
        "dense_activation": {"values": ["relu", "linear"]},
        "batch_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [5e-3, 5e-4, 5e-5]},
        "dropout": {"values": [0.3, 0.55, 0.8]},
        "optimizer": {"values": ["nadam", "adam", "sgd"]},
        "momentum": {"values": [0.8, 0.9, 0.98]},
        "beta_2": {"values": [0.9, 0.999, 0.99999]},
        "batch_normalization": {"values": [True, False]},
        "regularization": {"values": [5e-3, 1e-4, 1e-5, 0]},
        "savemodel": {"values": [False]},
        "latent_vars": {"values": [10]},
        "noise_std": {"values": [0.001, 0.01, 0.05, 0.1]}
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do hyperparameter \
optimization on slug flow dataset")
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

    if arg_dict['custom_config'] is not None:
        with open(arg_dict["custom_config"]) as json_file:
            cae_sweep_config = json.load(json_file)
    if arg_dict["savemodel"] == "True":
        cae_sweep_config['parameters']['savemodel'] = \
            {'values': [True]}

    cae_sweep_config['parameters']['datafile'] = \
        {'values': [arg_dict['datafile']]}

    sweep_id = wandb.sweep(Predictive_adversarial_sweep_config,
                           project='pred-aae', entity='zeff020')
    wandb.agent(sweep_id, train_wandb_pred_aae, count=arg_dict['niters'])
