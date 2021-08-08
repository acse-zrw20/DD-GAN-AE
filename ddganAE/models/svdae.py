"""

SVD autoencoder model. Can be used with any of the dense (or 1D convolutional)
encoder and decoder architectures in architectures directory.

"""

from keras.layers import Input, Conv1D
from keras.models import Model
import tensorflow as tf
import datetime
from ddganAE.utils import calc_pod, mse_weighted
import numpy as np
import wandb

__author__ = "Zef Wolffs"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


class SVDAE:
    """
    SVD Autoencoder class
    """

    def __init__(self, encoder, decoder, optimizer, seed=None):
        self.encoder = encoder
        self.decoder = decoder
        self.seed = seed
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
        coeffs, R, S = calc_pod(snapshots, nPOD, cumulative_tol)
        self.S = S  # Storing the singular values
        self.R = R
        return coeffs

    def reconstruct_from_pod(coeffs, R):
        return R @ coeffs

    def compile(self, nPOD, weight_loss=False):

        self.nPOD = nPOD
        if isinstance(self.encoder.layers[0], Conv1D):
            # Convolutional networks require a slightly different input shape
            self.input_shape = (1, nPOD)
        else:
            self.input_shape = (nPOD,)

        self.weight_loss = weight_loss

        vec = Input(shape=self.input_shape)
        encoded_repr = self.encoder(vec)
        gen_vec = self.decoder(encoded_repr)
        self.autoencoder = Model(vec, gen_vec)

        if not weight_loss:
            self.loss_f = None
            self.autoencoder.compile(optimizer=self.optimizer,
                                     loss='mse',
                                     metrics=['accuracy']
                                     )
        else:
            self.loss_f = mse_weighted()
            self.autoencoder.compile(optimizer=self.optimizer,
                                     loss=self.loss_f,
                                     metrics=['accuracy']
                                     )

    def train(self, train_data, epochs, val_data=None, batch_size=128,
              val_batch_size=128, wandb_log=False):

        loss_val = None
        # Returns POD as list of pod coefficients per subgrid
        coeffs = self.calc_pod(train_data, self.nPOD)

        if self.weight_loss:
            # Rescale
            self.loss_f.weights = np.interp(np.sqrt(self.S),
                                            (0, np.sqrt(self.S.max())),
                                            (0, +1))

        # Reshape to have multiple subgrids account for multiple batches
        out = np.zeros((coeffs[0].shape[0], len(coeffs)*coeffs[0].shape[-1]))
        for i, coeff in enumerate(coeffs):
            out[:, i*coeffs[0].shape[-1]:(i+1)*coeffs[0].shape[-1]] = coeff

        train_data = out.T

        if isinstance(self.encoder.layers[0], Conv1D):
            # Convolutional networks require a slightly different input shape
            train_data = np.expand_dims(train_data, 1)

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(buffer_size=train_data.shape[0],
                                              reshuffle_each_iteration=True,
                                              seed=self.seed).\
            batch(batch_size)

        if val_data is not None:

            # Reshape to have multiple subgrids account for multiple batches
            out = np.zeros((val_data[0].shape[0],
                            len(val_data)*val_data[0].shape[-1]))
            for i, coeff in enumerate(val_data):
                out[:, i*val_data[0].shape[-1]:(i+1)*val_data[0].shape[-1]] \
                    = coeff

            coeffs = []

            for iGrid in range(len(val_data)):
                snapshots_per_grid = \
                    out[:, iGrid*val_data[0].shape[-1]:(iGrid+1) *
                        val_data[0].shape[-1]]

                coeffs.append(np.dot(self.R.T, snapshots_per_grid))

            # Invert earlier operation of reshaping subgrids
            out = np.zeros((coeffs[0].shape[0],
                            len(coeffs)*coeffs[0].shape[-1]))
            for i, coeff in enumerate(coeffs):
                out[:, i*coeffs[0].shape[-1]:(i+1)*coeffs[0].shape[-1]] = coeff

            val_data = out.T

            if isinstance(self.encoder.layers[0], Conv1D):
                # Convolutional networks require a slightly different input
                # shape
                val_data = np.expand_dims(val_data, 1)

            val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
            val_dataset = val_dataset.shuffle(
                buffer_size=val_data.shape[0],
                reshuffle_each_iteration=True,
                seed=self.seed).\
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
            loss, acc = self.autoencoder.evaluate(val_grids, val_grids,
                                                  verbose=0)
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

    def predict(self, data):

        # Reshape to have multiple subgrids account for multiple batches
        out = np.zeros((data[0].shape[0],
                        len(data)*data[0].shape[-1]))
        for i, coeff in enumerate(data):
            out[:, i*data[0].shape[-1]:(i+1)*data[0].shape[-1]] \
                = coeff

        coeffs = []

        for iGrid in range(len(data)):
            snapshots_per_grid = \
                out[:, iGrid*data[0].shape[-1]:(iGrid+1) *
                    data[0].shape[-1]]

            coeffs.append(np.dot(self.R.T, snapshots_per_grid))

        # Invert earlier operation of reshaping subgrids
        out = np.zeros((coeffs[0].shape[0],
                        len(coeffs)*coeffs[0].shape[-1]))
        for i, coeff in enumerate(coeffs):
            out[:, i*coeffs[0].shape[-1]:(i+1)*coeffs[0].shape[-1]] = coeff

        val_data = out.T

        x_val_recon = self.autoencoder.predict(val_data)

        x_val_recon = \
            x_val_recon.reshape((len(val_data),
                                 int(x_val_recon.shape[0]/len(val_data)),
                                 -1))

        recon_grid = np.zeros(data.shape)
        for j in range(len(data)):
            recon = self.R @ x_val_recon[j].T
            recon_grid[j, :, :] = recon

        return recon_grid


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
