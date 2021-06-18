from keras.layers import Input
from keras.models import Model
import tensorflow as tf
import datetime
from ..utils import calc_pod
import numpy as np


class SVDAE:

    def __init__(self, encoder, decoder, optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = self.decoder.layers[0].input_shape[1]

        self.optimizer = optimizer

    def calc_pod(self, snapshots, nPOD=-2, cumulative_tol=0.99):
        """
        Calculate POD coefficients and basis functions

        Args:
            snapshots (list of ndarrays): List of arrays with subgrid
                                          snapshots. shape:
                                          (n_grids, n_nodes, n_timelevels)

        Returns:
            list of ndarrays: POD coefficients per subgrid
        """

        # Essentially just wraps a utility function
        coeffs, R = calc_pod(snapshots, nPOD, cumulative_tol)
        self.R = R
        return coeffs

    def reconstruct_from_pod(coeffs, R):
        return R @ coeffs

    def compile(self, nPOD):

        self.nPOD = nPOD
        self.input_shape = (nPOD,)

        vec = Input(shape=self.input_shape)
        encoded_repr = self.encoder(vec)
        gen_vec = self.decoder(encoded_repr)
        self.autoencoder = Model(vec, gen_vec)

        self.autoencoder.compile(optimizer=self.optimizer,
                                 loss='mse',
                                 metrics=['accuracy']
                                 )

    def train(self, train_data, epochs, val_data=None, batch_size=128,
              val_batch_size=128):

        loss_val = None
        # Returns POD as list of pod coefficients per subgrid
        coeffs = self.calc_pod(train_data, self.nPOD)

        # Reshape to have multiple subgrids account for multiple batches
        out = np.zeros((coeffs[0].shape[0], len(coeffs)*coeffs[0].shape[-1]))
        for i, coeff in enumerate(coeffs):
            out[:, i*coeffs[0].shape[-1]:(i+1)*coeffs[0].shape[-1]] = coeff

        train_data = out.T

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(buffer_size=train_data.shape[0],
                                              reshuffle_each_iteration=True).\
            batch(batch_size)

        if val_data is not None:
            coeffs = self.calc_pod(val_data, self.nPOD)

            # Reshape to have multiple subgrids account for multiple batches
            out = np.zeros((coeffs[0].shape[0],
                            len(coeffs)*coeffs[0].shape[-1]))
            for i, coeff in enumerate(coeffs):
                out[:, i*coeffs[0].shape[-1]:(i+1)*coeffs[0].shape[-1]] = coeff

            val_data = out.T

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
            acc = acc_cum/(step+1)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch)
                tf.summary.scalar('accuracy', acc, step=epoch)

            # Calculate the accuracies on the validation set
            if val_data is not None:
                loss_val, acc_val = self.validate(val_dataset)

                with val_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_val, step=epoch)
                    tf.summary.scalar('accuracy', acc_val, step=epoch)

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

    def predict_single(self, snapshot):
        coeff = (self.R.T@snapshot).reshape(1, -1)

        gen_coeff = self.autoencoder.predict(coeff)

        return self.R @ gen_coeff[0]


def print_losses(loss, epoch, loss_val=None):
    print("%d: [loss: %f, acc: %.2f%%]" %
          (epoch, loss[0], 100*loss[1]))

    if loss_val is not None:
        print("%d val: [loss: %f, acc: %.2f%%]" %
              (epoch, loss_val[0], 100*loss_val[1]))


def plot_losses(loss, liveloss, loss_val=None):

    if loss_val is not None:
        liveloss.update({'val_loss': loss[0],
                         'loss': loss_val[0]}
                        )
    else:
        liveloss.update({'loss': loss[0]})

    liveloss.send()
