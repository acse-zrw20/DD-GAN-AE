from keras.layers import Input
from keras.models import Model
import tensorflow as tf
import numpy as np
import datetime
import wandb


class CAE:

    def __init__(self, encoder, decoder, optimizer):
        self.encoder = encoder
        self.decoder = decoder
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

        self.autoencoder.compile(optimizer=self.optimizer,
                                 loss='mse',
                                 metrics=['accuracy'])

    def train(self, train_data, epochs, val_data=None, batch_size=128,
              val_batch_size=128, wandb_log=False):

        loss_val = None

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(buffer_size=train_data.shape[0],
                                              reshuffle_each_iteration=True).\
            batch(batch_size)

        if val_data is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
            val_dataset = val_dataset.shuffle(
                buffer_size=val_data.shape[0],
                reshuffle_each_iteration=True).\
                batch(val_batch_size)

        # Set up tensorboard logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        val_log_dir = 'logs/' + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for epoch in range(epochs):
            loss_cum = 0
            acc_cum = 0
            for step, grids in enumerate(train_dataset):

                # Train the autoencoder reconstruction
                loss, acc = self.autoencoder.train_on_batch(grids, grids)
                loss_cum += loss
                acc_cum += acc

            # Average the loss and accuracy over the entire dataset
            loss = loss_cum/(step+1)
            acc = acc_cum/step

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch)
                tf.summary.scalar('accuracy', acc, step=epoch)

            # Calculate the accuracies on the validation set
            if val_data is not None:
                loss_val, acc_val = self.validate(val_dataset)

                with val_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_val, step=epoch)
                    tf.summary.scalar('accuracy', acc_val, step=epoch)

            if wandb_log:
                if val_data is not None:
                    log = {"epoch": epoch, "train_loss": loss,
                           "train_accuracy": acc,
                           "valid_loss": loss_val,
                           "valid_accuracy": acc_val}
                else:
                    log = {"epoch": epoch, "train_loss": loss,
                           "train_accuracy": acc}

                wandb.log(log)

    def validate(self, val_dataset):

        loss_cum = 0
        acc_cum = 0
        for step, val_grids in enumerate(val_dataset):

            # Train the autoencoder reconstruction
            loss, acc = self.autoencoder.evaluate(val_grids, val_grids)
            loss_cum += loss
            acc_cum += acc

        # Average the loss and accuracy over the entire dataset
        loss = loss_cum/(step+1)
        acc = acc_cum/(step+1)

        return loss, acc

    def predict(self, data):

        return self.autoencoder.predict(data)
