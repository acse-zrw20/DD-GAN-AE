"""

Predictive Models

"""

from keras.layers import Input, Conv1D, GaussianNoise
from keras.models import Model
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
import wandb
import os


class Predictive_adversarial:
    """
    Predictive adversarial Autoencoder class
    """

    def __init__(self, encoder, decoder, discriminator, optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.latent_dim = self.decoder.layers[0].input_shape[1]

        self.optimizer = optimizer

    @classmethod
    def from_save(cls, dirname, optimizer):
        """
        Load model from savefile and override default constructor

        Args:
            dirname (str): Name of directory where model is saved
            optimizer (Object): Tensorflow optimizer
        """
        encoder = keras.models.load_model(dirname + '/encoder')
        decoder = keras.models.load_model(dirname + '/decoder')
        discriminator = keras.models.load_model(dirname +
                                                '/discriminator')

        return cls(encoder, decoder, discriminator, optimizer)

    def reconstruct_from_pod(coeffs, R):
        return R @ coeffs

    def compile(self, nPOD, increment=False):
        self.increment = increment
        self.nPOD = nPOD
        if isinstance(self.encoder.layers[0], Conv1D):
            # Convolutional networks require a slightly different input shape
            self.input_shape = (1, 3*nPOD)
        else:
            self.input_shape = (3*nPOD,)

        self.discriminator.compile(optimizer=self.optimizer,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])

        self.discriminator.trainable = False

        vec = Input(shape=self.input_shape)
        encoded_repr = self.encoder(vec)
        gen_vec = self.decoder(encoded_repr)

        valid = self.discriminator(encoded_repr)

        self.adversarial_autoencoder = Model(vec, [gen_vec, valid])

        self.adversarial_autoencoder.compile(loss=['mse',
                                                   'binary_crossentropy'],
                                             loss_weights=[0.999, 0.001],
                                             optimizer=self.optimizer)

    def preprocess(self, input_data):

        # Preprocessing
        for k in range(self.interval):
            grid_coeffs = np.array(input_data)[:, :, k::self.interval]
            train_data = np.zeros((grid_coeffs.shape[0] - 2,
                                   grid_coeffs.shape[1]
                                   * 3, grid_coeffs.shape[2]))

            for i in range(1, grid_coeffs.shape[0]-1):
                for j in range(3):
                    train_data[i-1, j*self.nPOD:(j+1)*self.nPOD, :] = \
                        grid_coeffs[i+j-1, :, :]

            train_data_swap = train_data.swapaxes(1, 2)

            step = train_data_swap[:, :-1, :]
            step_forward = train_data_swap[:, 1:, :]

            step[:, :, :self.nPOD] = step_forward[:, :, :self.nPOD]
            step[:, :, self.nPOD*2:] = step_forward[:, :, self.nPOD*2:]

            x_train = step

            if self.increment:
                # For if we use the time increment approach
                y_train = step
                y_train = np.diff(y_train, axis=1)
                x_train = x_train[:, :-1, :]
            else:
                y_train = step_forward

            x_train = np.concatenate(x_train, 0)
            y_train = np.concatenate(y_train, 0)

            # We only predict the central POD coefficients
            y_train = y_train[:, self.nPOD:self.nPOD*2]

            if k == 0:
                x_train_full = x_train
                y_train_full = y_train
            else:
                x_train_full = np.concatenate([x_train_full, x_train])
                y_train_full = np.concatenate([y_train_full, y_train])

        return x_train_full, y_train_full

    def train(self, input_data, epochs, interval=5, val_size=0,
              batch_size=128, val_batch_size=128, wandb_log=False,
              n_discriminator=5, n_gradient_ascent=np.inf, noise_std=0):
        """
        Training model where we use a training method that weights
        the losses of the discriminator and autoencoder and as such combines
        them into one loss and trains on them simultaneously. Takes in POD
        coefficients or latent variables. The timestep shifts will be done in
        this function.

        Args:
            train_data (np.ndarray): Array with train data, in shape
                                     (ngrids, nvars, ntime) (rescaled)
            epochs (int): Number of epochs
            interval (int): choose every `interval` variables to model
            val_data (np.ndarray, optional): Array with validation data.
                Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 128.
            plot_losses (bool, optional): Whether to plot losses.
                Defaults to False.
            print_losses (bool, optional): Whether to print losses.
                Defaults to False.
        """

        d_loss_val = g_loss_val = None
        self.interval = interval

        x_full, y_full = self.preprocess(input_data)

        if val_size > 0:
            x_train, x_val, y_train, y_val = train_test_split(
                x_full, y_full, test_size=val_size)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,
                                                            y_train))
        train_dataset = train_dataset.\
            shuffle(buffer_size=x_train.shape[0],
                    reshuffle_each_iteration=True).\
            batch(batch_size,
                  drop_remainder=True)

        if noise_std > 0:
            add_noise = tf.keras.Sequential([GaussianNoise(noise_std)])
            train_dataset = train_dataset.map(lambda x, y:
                                              (add_noise(float(x),
                                                         training=True), y))
        if val_size > 0:
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val,
                                                              y_val))
            val_dataset = val_dataset. \
                shuffle(buffer_size=x_val.shape[0],
                        reshuffle_each_iteration=True).\
                batch(val_batch_size,
                      drop_remainder=True)

        # Set up tensorboard logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        val_log_dir = 'logs/' + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Regularization phase
            d_loss_cum = 0
            g_loss_cum = 0
            g_step = 0
            step = 0
            for step, (x, y) in enumerate(train_dataset):

                latent_fake = self.encoder.predict(x)
                latent_real = np.random.normal(size=(batch_size,
                                                     self.latent_dim))

                if step % n_gradient_ascent == 0:
                    # Every so many timesteps do a step of gradient ascent to
                    # inhibit the discriminator from becoming too good
                    d_loss_real = self.discriminator.train_on_batch(
                        latent_real,
                        fake)[0]
                    d_loss_fake = self.discriminator.train_on_batch(
                        latent_fake,
                        valid)[0]
                    d_loss_cum += 0.5 * np.add(d_loss_real, d_loss_fake)
                else:
                    # Actually train the discriminator all of the other steps
                    d_loss_real = self.discriminator.train_on_batch(
                        latent_real,
                        valid)[0]
                    d_loss_fake = self.discriminator.train_on_batch(
                        latent_fake,
                        fake)[0]
                    d_loss_cum += 0.5 * np.add(d_loss_real, d_loss_fake)

                if step % n_discriminator == 0:

                    g_loss_cum += \
                        self.adversarial_autoencoder.train_on_batch(x,
                                                                    [y,
                                                                     valid])[0]
                    g_step += 1

            d_loss = d_loss_cum/(step+1)
            if g_step > 0:
                g_loss = g_loss_cum/(g_step)
            else:
                g_loss = 0

            with train_summary_writer.as_default():
                tf.summary.scalar('loss - g', g_loss, step=epoch)
                tf.summary.scalar('loss - d', d_loss, step=epoch)

            # Calculate the accuracies on the validation set
            if val_dataset is not None:

                d_loss_val, g_loss_val = self.validate(val_dataset,
                                                       val_batch_size)

                with val_summary_writer.as_default():
                    tf.summary.scalar('loss - g', g_loss_val, step=epoch)
                    tf.summary.scalar('loss - d', d_loss_val, step=epoch)

            if wandb_log:
                if val_dataset is not None:
                    log = {"epoch": epoch,
                           "g_train_loss": g_loss,
                           "d_train_loss": d_loss,
                           "g_valid_loss": g_loss_val,
                           "d_valid_loss": d_loss_val}
                else:
                    log = {"epoch": epoch,
                           "g_train_loss": g_loss,
                           "d_train_loss": d_loss}

                wandb.log(log)

    def validate(self, val_dataset, val_batch_size):
        """
        Validate model on previously unseen dataset.

        Args:
            val_dataset (np.array): Validation dataset
            val_batch_size (int, optional): Validation batch size. Defaults to
                                            128.

        Returns:
            tuple: Validation losses and accuracies
        """

        # Adversarial ground truths
        valid = np.ones((val_batch_size, 1))
        fake = np.zeros((val_batch_size, 1))

        d_loss_cum = 0
        g_loss_cum = 0
        step = 0
        for step, (x, y) in enumerate(val_dataset):

            latent_fake = self.encoder.predict(x)
            latent_real = np.random.normal(size=(val_batch_size,
                                                 self.latent_dim))

            d_loss_real = self.discriminator.evaluate(latent_real,
                                                      valid, verbose=0)[0]
            d_loss_fake = self.discriminator.evaluate(latent_fake,
                                                      fake, verbose=0)[0]
            d_loss_cum += 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss_cum += self.adversarial_autoencoder.evaluate(x,
                                                                [y,
                                                                 valid],
                                                                verbose=0)[0]

        # Average the loss and accuracy over the entire dataset
        d_loss = d_loss_cum/(step+1)
        g_loss = g_loss_cum/(step+1)

        return d_loss, g_loss

    def predict(self, boundaries, init_values, timesteps, iters=5, sor=1, 
                pre_interval=False):
        """
        Predict in time using boundaries and initial values for a certain
        number of timesteps. The timestep shifts will be done in this function

        Args:
            boundaries (np.ndarray): Boundaries in shape
                                     (nboundaries (2), nvars, ntimesteps)
            init_values (np.ndarray): Initial values in shape (ngrids, nvars)
            timesteps (int): Number of timesteps to predict
            iters (int): Number of iterations to do before a prediction
        """
        if pre_interval == False:
            boundaries = boundaries[:, :, ::self.interval]

        pred_vars = np.zeros((2 + init_values.shape[0], boundaries.shape[1],
                              boundaries.shape[2]))
        pred_vars[0] = boundaries[0]
        pred_vars[1:-1, :, 0] = init_values
        pred_vars[-1] = boundaries[1]

        for i in range(timesteps):
            # Outer "timesteps" loop
            
            # Let's start with a linear extrapolation for the predictions
            if i > 1:
                for k in range(1, init_values.shape[0]+1):
                    pred_vars[k, :, i+1] = pred_vars[k, :, i] + \
                                           (pred_vars[k, :, i] -
                                            pred_vars[k, :, i-1])

            for j in range(iters):
                # Inner optimization loop within a timestep
                for k in range(1, init_values.shape[0]+1):

                    # Loop over the columns that are meant to be predicted
                    # pred_vars[k, :, i+1] = \
                    #    self.adversarial_autoencoder.predict(
                    #        pred_vars[k-1:k+2, :, i].reshape(1, -1))[0]

                    pred_vars_t = \
                        np.concatenate((pred_vars[k-1, :, i+1],
                                        pred_vars[k, :, i],
                                        pred_vars[k+1, :, i+1])).reshape(1, -1)

                    if self.increment:
                        pred_vars[k, :, i+1] = pred_vars[k, :, i] + \
                            self.adversarial_autoencoder.predict(
                                pred_vars_t)[0]
                    else:
                        pred_vars[k, :, i+1] = pred_vars[k, :, i] + \
                            (self.adversarial_autoencoder.predict(
                                pred_vars_t)[0][0] - pred_vars[k, :, i]) * sor

                for k in range(init_values.shape[0], 0, -1):
                    # Loop over the columns that are meant to be predicted
                    pred_vars_t = \
                        np.concatenate((pred_vars[k-1, :, i+1],
                                        pred_vars[k, :, i],
                                        pred_vars[k+1, :, i+1])).reshape(1, -1)

                    if self.increment:
                        pred_vars[k, :, i+1] = pred_vars[k, :, i] + \
                            self.adversarial_autoencoder.predict(
                                pred_vars_t)[0]
                    else:
                        pred_vars[k, :, i+1] = pred_vars[k, :, i] + \
                            (self.adversarial_autoencoder.predict(
                                pred_vars_t)[0][0] - pred_vars[k, :, i]) * sor

        return pred_vars

    def save(self, dirname="model"):
        """
        Saves the model

        Args:
            dirname (str, optional): Directory to save model in. Defaults to
            "model"
        """

        os.mkdir(dirname)
        self.encoder.save(dirname + '/encoder')
        self.decoder.save(dirname + '/decoder')
        self.discriminator.save(dirname + '/discriminator')


class Predictive:
    """
    Predictive Autoencoder class
    """

    def __init__(self, encoder, decoder, optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = self.decoder.layers[0].input_shape[1]

        self.optimizer = optimizer

    def compile(self, nPOD, increment=False):
        """
        Compile model

        Args:
            input_shape (tuple): Shape of input data
        """

        self.increment = increment
        self.nPOD = nPOD
        if isinstance(self.encoder.layers[0], Conv1D):
            # Convolutional networks require a slightly different input shape
            self.input_shape = (1, 3*nPOD)
        else:
            self.input_shape = (3*nPOD,)

        grid = Input(shape=self.input_shape)
        encoded_repr = self.encoder(grid)
        gen_grid = self.decoder(encoded_repr)
        self.autoencoder = Model(grid, gen_grid)

        self.autoencoder.compile(optimizer=self.optimizer,
                                 loss="mse",
                                 metrics=['accuracy'])

    def preprocess(self, input_data):

        # Preprocessing
        for k in range(self.interval):
            grid_coeffs = np.array(input_data)[:, :, k::self.interval]
            train_data = np.zeros((grid_coeffs.shape[0] - 2,
                                   grid_coeffs.shape[1]
                                   * 3, grid_coeffs.shape[2]))

            for i in range(1, grid_coeffs.shape[0]-1):
                for j in range(3):
                    train_data[i-1, j*self.nPOD:(j+1)*self.nPOD, :] = \
                        grid_coeffs[i+j-1, :, :]

            train_data_swap = train_data.swapaxes(1, 2)

            step = train_data_swap[:, :-1, :]
            step_forward = train_data_swap[:, 1:, :]

            step[:, :, :self.nPOD] = step_forward[:, :, :self.nPOD]
            step[:, :, self.nPOD*2:] = step_forward[:, :, self.nPOD*2:]

            x_train = step

            if self.increment:
                # For if we use the time increment approach
                y_train = step
                y_train = np.diff(y_train, axis=1)
                x_train = x_train[:, :-1, :]
            else:
                y_train = step_forward

            x_train = np.concatenate(x_train, 0)
            y_train = np.concatenate(y_train, 0)

            # We only predict the central POD coefficients
            y_train = y_train[:, self.nPOD:self.nPOD*2]

            if k == 0:
                x_train_full = x_train
                y_train_full = y_train
            else:
                x_train_full = np.concatenate([x_train_full, x_train])
                y_train_full = np.concatenate([y_train_full, y_train])

        return x_train_full, y_train_full

    def train(self, input_data, epochs, interval=5, val_size=0,
              batch_size=128, val_batch_size=128, wandb_log=False,
              n_discriminator=5, n_gradient_ascent=np.inf, noise_std=0):
        """
        Training model where we use a training method that weights
        the losses of the discriminator and autoencoder and as such combines
        them into one loss and trains on them simultaneously. Takes in POD
        coefficients or latent variables. The timestep shifts will be done in
        this function.

        Args:
            train_data (np.ndarray): Array with train data, in shape
                                     (ngrids, nvars, ntime) (rescaled)
            epochs (int): Number of epochs
            interval (int): choose every `interval` variables to model
            val_data (np.ndarray, optional): Array with validation data.
                Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 128.
            plot_losses (bool, optional): Whether to plot losses.
                Defaults to False.
            print_losses (bool, optional): Whether to print losses.
                Defaults to False.
        """

        val_dataset = None
        self.interval = interval

        x_full, y_full = self.preprocess(input_data)

        if val_size > 0:
            x_train, x_val, y_train, y_val = train_test_split(
                x_full, y_full, test_size=val_size)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,
                                                            y_train))
        train_dataset = train_dataset.\
            shuffle(buffer_size=x_train.shape[0],
                    reshuffle_each_iteration=True).\
            batch(batch_size,
                  drop_remainder=True)

        if noise_std > 0:
            add_noise = tf.keras.Sequential([GaussianNoise(noise_std)])
            train_dataset = train_dataset.map(lambda x, y:
                                              (add_noise(float(x),
                                                         training=True), y))
        if val_size > 0:
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val,
                                                              y_val))
            val_dataset = val_dataset. \
                shuffle(buffer_size=x_val.shape[0],
                        reshuffle_each_iteration=True).\
                batch(val_batch_size,
                      drop_remainder=True)

        # Set up tensorboard logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        val_log_dir = 'logs/' + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for epoch in range(epochs):
            loss_cum = 0
            acc_cum = 0
            for step, (x, y) in enumerate(train_dataset):

                # Train the autoencoder reconstruction
                loss, acc = self.autoencoder.train_on_batch(x, y)
                loss_cum += loss
                acc_cum += acc

            # Average the loss and accuracy over the entire dataset
            loss = loss_cum/(step+1)
            acc = acc_cum/step

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch)
                tf.summary.scalar('accuracy', acc, step=epoch)

            # Calculate the accuracies on the validation set
            if val_dataset is not None:
                loss_val, acc_val = self.validate(val_dataset, val_batch_size)

                with val_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_val, step=epoch)
                    tf.summary.scalar('accuracy', acc_val, step=epoch)

            if wandb_log:
                if val_dataset is not None:
                    log = {"epoch": epoch, "train_loss": loss,
                           "train_accuracy": acc,
                           "valid_loss": loss_val,
                           "valid_accuracy": acc_val}
                else:
                    log = {"epoch": epoch, "train_loss": loss,
                           "train_accuracy": acc}

                wandb.log(log)

    def validate(self, val_dataset, val_batch_size):
        """
        Validate model on previously unseen dataset.

        Args:
            val_dataset (np.array): Validation dataset
            val_batch_size (int, optional): Validation batch size. Defaults to
                                            128.

        Returns:
            tuple: Validation losses and accuracies
        """

        loss_cum = 0
        acc_cum = 0
        step = 0
        for step, (x, y) in enumerate(val_dataset):

            # Train the autoencoder reconstruction
            loss, acc = self.autoencoder.evaluate(x, y,
                                                  verbose=0)
            loss_cum += loss
            acc_cum += acc

        # Average the loss and accuracy over the entire dataset
        loss = loss_cum/(step+1)
        acc = acc_cum/(step+1)

        return loss, acc

    def predict(self, boundaries, init_values, timesteps, iters=5, sor=1):
        """
        Predict in time using boundaries and initial values for a certain
        number of timesteps. The timestep shifts will be done in this function

        Args:
            boundaries (np.ndarray): Boundaries in shape
                                     (nboundaries (2), nvars, ntimesteps)
            init_values (np.ndarray): Initial values in shape (ngrids, nvars)
            timesteps (int): Number of timesteps to predict
            iters (int): Number of iterations to do before a prediction
        """
        boundaries = boundaries[:, :, ::self.interval]

        pred_vars = np.zeros((2 + init_values.shape[0], boundaries.shape[1],
                              boundaries.shape[2]))
        pred_vars[0] = boundaries[0]
        pred_vars[1:-1, :, 0] = init_values
        pred_vars[-1] = boundaries[1]

        for i in range(timesteps):
            # Outer "timesteps" loop
            
            # Let's start with a linear extrapolation for the predictions
            if i > 1:
                for k in range(1, init_values.shape[0]+1):
                    pred_vars[k, :, i+1] = pred_vars[k, :, i] + \
                                           (pred_vars[k, :, i] -
                                            pred_vars[k, :, i-1])

            for j in range(iters):
                # Inner optimization loop within a timestep
                for k in range(1, init_values.shape[0]+1):

                    # Loop over the columns that are meant to be predicted
                    # pred_vars[k, :, i+1] = \
                    #    self.adversarial_autoencoder.predict(
                    #        pred_vars[k-1:k+2, :, i].reshape(1, -1))[0]

                    pred_vars_t = \
                        np.concatenate((pred_vars[k-1, :, i+1],
                                        pred_vars[k, :, i],
                                        pred_vars[k+1, :, i+1])).reshape(1, -1)

                    if self.increment:
                        pred_vars[k, :, i+1] = pred_vars[k, :, i] + \
                            self.autoencoder.predict(
                                pred_vars_t)
                    else:
                        pred_vars[k, :, i+1] = pred_vars[k, :, i] + \
                            (self.autoencoder.predict(
                                pred_vars_t)[0] - pred_vars[k, :, i]) * sor

                for k in range(init_values.shape[0], 0, -1):
                    # Loop over the columns that are meant to be predicted
                    pred_vars_t = \
                        np.concatenate((pred_vars[k-1, :, i+1],
                                        pred_vars[k, :, i],
                                        pred_vars[k+1, :, i+1])).reshape(1, -1)

                    if self.increment:
                        pred_vars[k, :, i+1] = pred_vars[k, :, i] + \
                            self.autoencoder.predict(
                                pred_vars_t)
                    else:
                        pred_vars[k, :, i+1] = pred_vars[k, :, i] + \
                            (self.autoencoder.predict(
                                pred_vars_t)[0] - pred_vars[k, :, i]) * sor

        return pred_vars
