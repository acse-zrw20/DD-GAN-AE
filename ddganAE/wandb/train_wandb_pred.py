"""

Functions used for weights and biases hyperparameter optimization on slug flow
dataset

"""

import wandb
import tensorflow as tf
import argparse
import os
import json
import keras
from sklearn.preprocessing import MinMaxScaler
from ddganAE.models import Predictive_adversarial, Predictive
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
from ddganAE.wandb.get_snapshots_3d_continuous import \
     get_snapshots_3D
import numpy as np

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
    with wandb.init(config=config, tags=["central_doms_pred_mse"]):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Data processing
        latent_vars = np.load(config.datafile)

        nfiles = int(latent_vars.shape[0]/config.domains)

        latent_vars_reshaped = np.moveaxis(
            latent_vars.reshape(nfiles, config.domains, config.in_vars),
            0, 2)

        train_data = latent_vars_reshaped[:config.domains]

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
        pred_adv.compile(config.in_vars, increment=config.increment)
        pred_adv.train(
            train_data,
            config.epochs,
            interval=config.interval,
            batch_size=config.batch_size,
            val_size=0.1,
            wandb_log=True,
            noise_std=config.noise_std,
            n_discriminator=config.n_discriminator,
            n_gradient_ascent=config.n_gradient_ascent
        )

        # Check how well the model actually performs to also predict the
        # results

        # Create boundaries and initial values arrays for prediction later
        boundaries = np.zeros((2, config.in_vars, nfiles))
        boundaries[0] = train_data[4]
        boundaries[1] = train_data[7]

        init_values = np.zeros((2, config.in_vars))
        init_values[0] = train_data[5][:, 0]
        init_values[1] = train_data[6][:, 0]

        predicted = pred_adv.predict(boundaries, init_values,
                                     int(nfiles/config.interval)-1, iters=5)
        train_data_int = train_data[4:8, :, ::config.interval]

        mse = tf.keras.losses.MeanSquaredError()
        mse_pred = mse(predicted[:, :, :int(nfiles/config.interval)-2],
                       train_data_int[:, :, :int(nfiles/config.interval)-2])\
            .numpy()

        log = {"prediction_mse": mse_pred}

        wandb.log(log)

        if config.savemodel:
            dirname = "model_" + wandb.run.name
            os.mkdir(dirname)
            pred_adv.encoder.save(dirname + '/encoder')
            pred_adv.decoder.save(dirname + '/decoder')


def train_wandb_pred_ae(config=None):
    """
    Construct and subsequently train the model while reporting losses to
    weights and biases platform. Weights and biases also controls
    hyperparameters.

    Args:
        config (dict, optional): Dictionary with hyperparameters, set by
                                 weights and biases. Defaults to None.
    """
    with wandb.init(config=config, tags=["central_doms_pred_mse"]):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Data processing
        latent_vars = np.load(config.datafile)

        nfiles = int(latent_vars.shape[0]/config.domains)

        latent_vars_reshaped = np.moveaxis(
            latent_vars.reshape(nfiles, config.domains, config.in_vars),
            0, 2)

        train_data = latent_vars_reshaped[:config.domains]

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

        pred_adv = Predictive(encoder, decoder,
                              optimizer)
        pred_adv.compile(config.in_vars, increment=config.increment)
        pred_adv.train(
            train_data,
            config.epochs,
            interval=config.interval,
            batch_size=config.batch_size,
            val_size=0.1,
            wandb_log=True,
            noise_std=config.noise_std
        )

        # Check how well the model actually performs to also predict the
        # results

        # Create boundaries and initial values arrays for prediction later
        boundaries = np.zeros((2, config.in_vars, nfiles))
        boundaries[0] = train_data[4]
        boundaries[1] = train_data[7]

        init_values = np.zeros((2, config.in_vars))
        init_values[0] = train_data[5][:, 0]
        init_values[1] = train_data[6][:, 0]

        predicted = pred_adv.predict(boundaries, init_values,
                                     int(nfiles/config.interval)-1, iters=5)
        train_data_int = train_data[4:8, :, ::config.interval]

        mse = tf.keras.losses.MeanSquaredError()
        mse_pred = mse(predicted[:, :, :int(nfiles/config.interval)-2],
                       train_data_int[:, :, :int(nfiles/config.interval)-2])\
            .numpy()

        log = {"prediction_mse": mse_pred}

        wandb.log(log)

        if config.savemodel:
            dirname = "model_" + wandb.run.name
            os.mkdir(dirname)
            pred_adv.encoder.save(dirname + '/encoder')
            pred_adv.decoder.save(dirname + '/decoder')


def continuous_train_wandb_pred_aae(config=None):
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
        pred_adv.compile(config.in_vars, increment=config.increment)

        nfiles = 800

        for i in range(config.n_epochs):

            # Data processing
            grids = get_snapshots_3D(nfiles=nfiles,
                                     ndomains=config.domains,
                                     in_file_base=config.datafile,
                                     save=False)

            # Set all >0 to 0 and all <1 to 1 for alpha field
            grids[:, :, :, :, 3][np.where(grids[:, :, :, :, 3] < 0)] = 0
            grids[:, :, :, :, 3][np.where(grids[:, :, :, :, 3] > 1)] = 1

            # Rescale all the velocity fields
            scaler = MinMaxScaler()
            grids[:, :, :, :, :3] = scaler.fit_transform(grids[:, :, :, :, :3]
                                                         .reshape(-1, 1))\
                .reshape(grids[:, :, :, :, :3].shape)

            initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                             stddev=0.05,
                                                             seed=None)
            optimizer = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.98,
                                                 beta_2=0.9)

            encoder = keras.models.load_model(config.encoder_folder +
                                              "/encoder")

            latent_vars = encoder.predict(grids)

            latent_vars_reshaped = np.moveaxis(
                latent_vars.reshape(nfiles, config.domains, config.in_vars),
                0, 2)

            train_data = latent_vars_reshaped

            # Scaling the latent variables
            scaler = MinMaxScaler((-1, 1))
            train_data = scaler.fit_transform(
                train_data.reshape(-1, 1)).reshape(train_data.shape)

            # Generate a new set of training data every n epochs
            pred_adv.train(
                train_data,
                config.epochs,
                interval=config.interval,
                batch_size=config.batch_size,
                val_size=0.1,
                wandb_log=True,
                noise_std=config.noise_std,
                n_discriminator=config.n_discriminator,
                n_gradient_ascent=config.n_gradient_ascent
            )

            # Check how well the model actually performs to also predict the
            # results

            # Create boundaries and initial values arrays for prediction later
            boundaries = np.zeros((2, config.in_vars, nfiles))
            boundaries[0] = train_data[0]
            boundaries[1] = train_data[3]

            init_values = np.zeros((2, 10))
            init_values[0] = train_data[1][:, 0]
            init_values[1] = train_data[2][:, 0]

            predicted = pred_adv.predict(boundaries, init_values,
                                         int(nfiles/config.interval)-1,
                                         iters=5)
            train_data_int = train_data[:, :, ::config.interval]

            mse = tf.keras.losses.MeanSquaredError()
            mse_pred = mse(predicted[:, :, :int(nfiles/config.interval)-2],
                           train_data_int[:4, :, :int(nfiles/config.interval) -
                           2])\
                .numpy()

            log = {"prediction_mse": mse_pred}

            wandb.log(log)

        if config.savemodel:
            dirname = "model_" + wandb.run.name
            os.mkdir(dirname)
            pred_adv.encoder.save(dirname + '/encoder')
            pred_adv.decoder.save(dirname + '/decoder')


# Configuration options for hyperparameter optimization
Predictive_adversarial_sweep_config = {
    "method": "bayes",
    "metric": {"name": "prediction_mse", "goal": "minimize"},
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
        "in_vars": {"values": [100]},
        "dense_activation": {"values": ["relu", "linear"]},
        "batch_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [5e-3, 5e-4, 5e-5]},
        "dropout": {"values": [0.3, 0.55, 0.8]},
        "optimizer": {"values": ["nadam", "adam", "sgd"]},
        "momentum": {"values": [0.8, 0.9, 0.98]},
        "beta_2": {"values": [0.9, 0.999, 0.99999]},
        "batch_normalization": {"values": [True, False]},
        "regularization": {"values": [1e-3, 1e-4, 1e-5, 1e-6, 0]},
        "savemodel": {"values": [False]},
        "latent_vars": {"values": [100, 300, 500]},
        "interval": {"values": [1, 2, 4, 6]},
        "final_act": {
            "values": [
              "linear",
              "sigmoid",
              "tanh"
            ]
        },
        "noise_std": {"values": [0.00001, 0.001, 0.01, 0.05, 0.1]},
        "increment": {"values": [True, False]},
        "epochs": {"values": [200, 500, 1000, 2000]},
        "n_discriminator": {"values": [1, 2, 4, 5]},
        "n_gradient_ascent": {"values": [3, 8, 15, 30]},
        "domains": {"values": [10]}
    },
}

# Configuration options for hyperparameter optimization
Predictive_ae_sweep_config = {
    "method": "bayes",
    "metric": {"name": "prediction_mse", "goal": "minimize"},
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
        "in_vars": {"values": [100]},
        "dense_activation": {"values": ["relu", "linear"]},
        "batch_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [5e-3, 5e-4, 5e-5]},
        "dropout": {"values": [0.3, 0.55, 0.8]},
        "optimizer": {"values": ["nadam", "adam", "sgd"]},
        "momentum": {"values": [0.8, 0.9, 0.98]},
        "beta_2": {"values": [0.9, 0.999, 0.99999]},
        "batch_normalization": {"values": [True, False]},
        "regularization": {"values": [1e-3, 1e-4, 1e-5, 1e-6, 0]},
        "savemodel": {"values": [False]},
        "latent_vars": {"values": [60, 100, 300]},
        "interval": {"values": [1, 2, 4, 6]},
        "final_act": {
            "values": [
              "linear",
              "sigmoid",
              "tanh"
            ]
        },
        "noise_std": {"values": [0.00001, 0.001, 0.01, 0.05, 0.1]},
        "increment": {"values": [True, False]},
        "epochs": {"values": [100, 200, 500, 1000]},
        "domains": {"values": [10]}
    },
}

# Configuration options for hyperparameter optimization
Continuous_predictive_adversarial_sweep_config = {
    "method": "random",
    "metric": {"name": "prediction_mse", "goal": "minimize"},
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
        "regularization": {"values": [1e-3, 1e-4, 1e-5, 1e-6, 0]},
        "savemodel": {"values": [False]},
        "latent_vars": {"values": [30, 50, 100]},
        "interval": {"values": [1, 2, 4, 6]},
        "final_act": {
            "values": [
              "linear",
              "sigmoid",
              "tanh"
            ]
        },
        "noise_std": {"values": [0.00001, 0.001, 0.01, 0.05, 0.1]},
        "increment": {"values": [True, False]},
        "epochs": {"values": [50, 100, 150]},
        "n_discriminator": {"values": [1]},
        "n_gradient_ascent": {"values": [3, 8, 15, 30]},
        "domains": {"values": [6]},
        "n_epochs": {"values": [20]}
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do hyperparameter \
optimization on slug flow dataset")
    parser.add_argument('--datafile', type=str, nargs='?',
                        default="/home/zef/Documents/master/acse-9/DD-GAN-AE/\
submodules/DD-GAN/data/processed/cae_latent_sf_10vars_800steps_different.npy",
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
    parser.add_argument('--continuous', action='store_true', default=False, 
                        help='whether to use continuous learning \
functionality')
    parser.add_argument('--encoder_folder', type=str, nargs='?',
                        default=None,
                        help='folder with autoencoder for generating latent \
variables')
    parser.add_argument('--model', type=str, nargs='?',
                        default=None,
                        help='Choose either ae (normal autoencoder) or aae \
(adversarial autoencoder)')
    args = parser.parse_args()

    arg_dict = vars(args)

    if args.continuous:
        if arg_dict['custom_config'] is not None:
            with open(arg_dict["custom_config"]) as json_file:
                Continuous_predictive_adversarial_sweep_config = \
                    json.load(json_file)
        if arg_dict["savemodel"] == "True":
            Continuous_predictive_adversarial_sweep_config['parameters'][
                'savemodel'] = \
                {'values': [True]}

        Continuous_predictive_adversarial_sweep_config['parameters'][
            'datafile'] = \
            {'values': [arg_dict['datafile']]}

        Continuous_predictive_adversarial_sweep_config['parameters'][
            'encoder_folder'] = \
            {'values': [arg_dict['encoder_folder']]}

        sweep_id = wandb.sweep(Continuous_predictive_adversarial_sweep_config,
                               project='pred-aae', entity='zeff020')
        wandb.agent(sweep_id, continuous_train_wandb_pred_aae, 
                    count=arg_dict['niters'])
    if args.model == "ae":
        # Use the normal autoencoder for predictions
        if arg_dict['custom_config'] is not None:
            with open(arg_dict["custom_config"]) as json_file:
                Predictive_ae_sweep_config = json.load(json_file)
        if arg_dict["savemodel"] == "True":
            Predictive_ae_sweep_config['parameters']['savemodel'] = \
                {'values': [True]}

        Predictive_ae_sweep_config['parameters']['datafile'] = \
            {'values': [arg_dict['datafile']]}

        sweep_id = wandb.sweep(Predictive_ae_sweep_config,
                               project='pred-ae', entity='zeff020')
        wandb.agent(sweep_id, train_wandb_pred_ae, count=arg_dict['niters'])
    elif args.model == "aae":
        # Use the adversarial autoencoder for predictions
        if arg_dict['custom_config'] is not None:
            with open(arg_dict["custom_config"]) as json_file:
                Predictive_adversarial_sweep_config = json.load(json_file)
        if arg_dict["savemodel"] == "True":
            Predictive_adversarial_sweep_config['parameters']['savemodel'] = \
                {'values': [True]}

        Predictive_adversarial_sweep_config['parameters']['datafile'] = \
            {'values': [arg_dict['datafile']]}

        sweep_id = wandb.sweep(Predictive_adversarial_sweep_config,
                               project='pred-aae', entity='zeff020')
        wandb.agent(sweep_id, train_wandb_pred_aae, count=arg_dict['niters'])
