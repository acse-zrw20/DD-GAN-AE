from keras.layers import Input
from keras.models import Model
import numpy as np
import tensorflow as tf
import datetime
import wandb


class AAE:

    def __init__(self, encoder, decoder, discriminator, optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.latent_dim = self.decoder.layers[0].input_shape[1]

        self.optimizer = optimizer

    def compile(self, input_shape):
        """
        Compilation of models according to original paper on adversarial
        autoencoders

        Args:
            input_shape (tuple): Shape of input data
        """

        self.input_shape = input_shape

        grid = Input(shape=self.input_shape)
        encoded_repr = self.encoder(grid)
        gen_grid = self.decoder(encoded_repr)
        self.autoencoder = Model(grid, gen_grid)

        valid = self.discriminator(encoded_repr)
        self.encoder_discriminator = Model(grid, valid)

        self.discriminator.compile(optimizer=self.optimizer,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])

        self.autoencoder.compile(optimizer=self.optimizer,
                                 loss='mse',
                                 metrics=['accuracy'])

        self.discriminator.trainable = False

        self.encoder_discriminator.compile(optimizer=self.optimizer,
                                           loss='binary_crossentropy',
                                           metrics=['accuracy'])

    def train(self, train_data, epochs, val_data=None, batch_size=128,
              val_batch_size=128, wandb_log=False):
        """
        Training model according to original paper on adversarial autoencoders

        Args:
            train_data (np.ndarray): Array with train data
            epochs (int): Number of epochs
            val_data (np.ndarray, optional): Array with validation data.
                Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 128.
            plot_losses (bool, optional): Whether to plot losses.
                Defaults to False.
            print_losses (bool, optional): Whether to print losses.
                Defaults to False.
        """

        d_loss_val = g_loss_val = None

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(buffer_size=train_data.shape[0],
                                              reshuffle_each_iteration=True).\
            batch(batch_size, drop_remainder=True)

        if val_data is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
            val_dataset = val_dataset.shuffle(
                buffer_size=val_data.shape[0],
                reshuffle_each_iteration=True).\
                batch(val_batch_size, drop_remainder=True)

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

            # Reconstruction phase
            loss_cum = 0
            acc_cum = 0
            for step, grids in enumerate(train_dataset):
                # Train the autoencoder reconstruction
                loss, acc = self.autoencoder.train_on_batch(grids, grids)
                loss_cum += loss
                acc_cum += acc

            # Average the loss and accuracy over the entire dataset
            loss = loss_cum/(step+1)
            acc = acc_cum/(step+1)

            # Regularization phase
            d_loss_cum = 0
            g_loss_cum = 0
            for step, grids in enumerate(train_dataset):

                # Generate real and fake latent space. Fake latent space is
                # the normal distribution
                latent_fake = self.encoder.predict(grids)
                latent_real = np.random.normal(size=(batch_size,
                                                     self.latent_dim))

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(latent_real,
                                                                valid)[0]
                d_loss_fake = self.discriminator.train_on_batch(latent_fake,
                                                                fake)[0]
                d_loss_cum += 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train generator
                g_loss_cum += \
                    self.encoder_discriminator.train_on_batch(grids, valid)[0]

            d_loss = d_loss_cum/(step+1)
            g_loss = g_loss_cum/(step+1)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss - ae', loss, step=epoch)
                tf.summary.scalar('accuracy - ae', acc, step=epoch)
                tf.summary.scalar('loss - g', g_loss, step=epoch)
                tf.summary.scalar('loss - d', d_loss, step=epoch)

            # Calculate the accuracies on the validation set
            if val_data is not None:
                loss_val, acc_val, d_loss_val, g_loss_val = \
                    self.validate(val_dataset, val_batch_size)

                with val_summary_writer.as_default():
                    tf.summary.scalar('loss - ae', loss_val, step=epoch)
                    tf.summary.scalar('accuracy - ae', acc_val, step=epoch)
                    tf.summary.scalar('loss - g', g_loss_val, step=epoch)
                    tf.summary.scalar('loss - d', d_loss_val, step=epoch)

            if wandb_log:
                if val_data is not None:
                    log = {"epoch": epoch, "train_loss": loss,
                           "train_accuracy": acc,
                           "g_train_loss": g_loss,
                           "d_train_loss": d_loss,
                           "g_valid_loss": g_loss_val,
                           "d_valid_loss": d_loss_val,
                           "valid_loss": loss_val,
                           "valid_accuracy": acc_val}
                else:
                    log = {"epoch": epoch, "train_loss": loss,
                           "train_accuracy": acc,
                           "g_train_loss": g_loss,
                           "d_train_loss": d_loss}

                wandb.log(log)

    def validate(self, val_dataset, val_batch_size=128):

        # Adversarial ground truths
        valid = np.ones((val_batch_size, 1))
        fake = np.zeros((val_batch_size, 1))

        loss_cum = 0
        acc_cum = 0
        d_loss_cum = 0
        g_loss_cum = 0
        for step, val_grids in enumerate(val_dataset):

            loss, acc = self.autoencoder.evaluate(val_grids, val_grids, 
                                                  verbose=0)
            loss_cum += loss
            acc_cum += acc

            latent_fake = self.encoder.predict(val_grids)
            latent_real = np.random.normal(size=(val_batch_size, 
                                                 self.latent_dim))

            d_loss_real = self.discriminator.evaluate(latent_real,
                                                      valid, verbose=0)[0]
            d_loss_fake = self.discriminator.evaluate(latent_fake,
                                                      fake, verbose=0)[0]
            d_loss_cum += 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss_cum += \
                self.encoder_discriminator.evaluate(val_grids, valid,
                                                    verbose=0)[0]

        # Average the loss and accuracy over the entire dataset
        loss = loss_cum/(step+1)
        acc = acc_cum/(step+1)
        d_loss = d_loss_cum/(step+1)
        g_loss = g_loss_cum/(step+1)

        return loss, acc, d_loss, g_loss


class AAE_combined_loss:

    def __init__(self, encoder, decoder, discriminator, optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.latent_dim = self.decoder.layers[0].input_shape[1]

        self.optimizer = optimizer

    def compile(self, input_shape):
        """
        Compilation of models where we use a training method that weights
        the losses of the discriminator and autoencoder and as such combines
        them into one loss and trains on them simultaneously.

        Args:
            input_shape (tuple): Shape of input data
        """

        self.input_shape = input_shape

        self.discriminator.compile(optimizer=self.optimizer,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])

        grid = Input(shape=self.input_shape)
        encoded_repr = self.encoder(grid)
        reconstructed_grid = self.decoder(encoded_repr)

        self.discriminator.trainable = False

        valid = self.discriminator(encoded_repr)

        self.adversarial_autoencoder = Model(grid, [reconstructed_grid, valid])

        self.adversarial_autoencoder.compile(loss=['mse',
                                                   'binary_crossentropy'],
                                             loss_weights=[0.999, 0.001],
                                             optimizer=self.optimizer)

    def train(self, train_data, epochs, val_data=None,
              batch_size=128, val_batch_size=128, wandb_log=False):
        """
        Training model where we use a training method that weights
        the losses of the discriminator and autoencoder and as such combines
        them into one loss and trains on them simultaneously.

        Args:
            train_data (np.ndarray): Array with train data
            epochs (int): Number of epochs
            val_data (np.ndarray, optional): Array with validation data.
                Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 128.
            plot_losses (bool, optional): Whether to plot losses.
                Defaults to False.
            print_losses (bool, optional): Whether to print losses.
                Defaults to False.
        """

        d_loss_val = g_loss_val = None

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(buffer_size=train_data.shape[0],
                                              reshuffle_each_iteration=True).\
            batch(batch_size, drop_remainder=True)

        if val_data is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
            val_dataset = val_dataset.shuffle(
                buffer_size=val_data.shape[0],
                reshuffle_each_iteration=True).\
                batch(val_batch_size, drop_remainder=True)

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
            for step, grids in enumerate(train_dataset):

                latent_fake = self.encoder.predict(grids)
                latent_real = np.random.normal(size=(batch_size, 
                                                     self.latent_dim))

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(latent_real,
                                                                valid)[0]
                d_loss_fake = self.discriminator.train_on_batch(latent_fake,
                                                                fake)[0]
                d_loss_cum += 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the generator
                g_loss_cum += \
                    self.adversarial_autoencoder.train_on_batch(grids,
                                                                [grids,
                                                                 valid])[0]

            d_loss = d_loss_cum/(step+1)
            g_loss = g_loss_cum/(step+1)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss - g', g_loss, step=epoch)
                tf.summary.scalar('loss - d', d_loss, step=epoch)

            # Calculate the accuracies on the validation set
            if val_data is not None:
                d_loss_val, g_loss_val = self.validate(val_dataset,
                                                       val_batch_size)

                with val_summary_writer.as_default():
                    tf.summary.scalar('loss - g', g_loss_val, step=epoch)
                    tf.summary.scalar('loss - d', d_loss_val, step=epoch)

    def validate(self, val_dataset, val_batch_size=128):

        # Adversarial ground truths
        valid = np.ones((val_batch_size, 1))
        fake = np.zeros((val_batch_size, 1))

        d_loss_cum = 0
        g_loss_cum = 0
        for step, val_grids in enumerate(val_dataset):

            latent_fake = self.encoder.predict(val_grids)
            latent_real = np.random.normal(size=(val_batch_size,
                                                 self.latent_dim))

            d_loss_real = self.discriminator.evaluate(latent_real,
                                                      valid)[0]
            d_loss_fake = self.discriminator.evaluate(latent_fake,
                                                      fake)[0]
            d_loss_cum += 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss_cum += self.adversarial_autoencoder.evaluate(val_grids,
                                                                [val_grids,
                                                                 valid])[0]

        # Average the loss and accuracy over the entire dataset
        d_loss = d_loss_cum/(step+1)
        g_loss = g_loss_cum/(step+1)

        return d_loss, g_loss


def print_losses(d_loss, g_loss, epoch, d_loss_val=None, g_loss_val=None):
    print("%d: [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" %
          (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

    if d_loss_val is not None and g_loss_val is not None:
        print("%d val: [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" %
              (epoch, d_loss_val[0], 100*d_loss_val[1], g_loss_val[0],
               g_loss_val[1]))


def plot_losses(d_loss, g_loss, liveloss, d_loss_val=None,
                g_loss_val=None):

    if d_loss_val is not None and g_loss_val is not None:
        liveloss.update({'val_generator_loss_training': g_loss[0],
                         'generator_loss_validation': g_loss_val[0],
                         'discriminator_loss_training': d_loss[0],
                         'val_discriminator_loss_validation':
                         d_loss_val[0]}
                        )
    else:
        liveloss.update({'generator_loss_training': g_loss[0],
                         'discriminator_loss_training': d_loss[0]})

    liveloss.send()
