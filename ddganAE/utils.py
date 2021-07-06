"""

General utilities for package

"""

import numpy as np
import keras.backend as K
from keras.losses import mse

__author__ = "Zef Wolffs"
__credits__ = ["Claire Heaney"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


def calc_pod(snapshots, nPOD=-2, cumulative_tol=0.99):
    """
    Calculate POD coefficients and basis functions

    Args:
        snapshots (list of ndarrays): List of arrays with subgrid
                                        snapshots. shape:
                                        (n_grids or n_scalar, n_nodes, 
                                        n_timelevels)

    Returns:
        list of ndarrays: POD coefficients per subgrid
    """

    # Reshape to have multiple subgrids account for multiple batches
    out = np.zeros((snapshots[0].shape[0],
                    len(snapshots)*snapshots[0].shape[-1]))
    for i, coeff in enumerate(snapshots):
        out[:, i*snapshots[0].shape[-1]:(i+1)*snapshots[0].shape[-1]] = coeff

    u, s, v = np.linalg.svd(out)

    cumulative_info = np.zeros(len(s))
    for j in range(len(s)):
        if j == 0:
            cumulative_info[j] = s[j]
        else:
            cumulative_info[j] = cumulative_info[j-1] + s[j]

    cumulative_info = cumulative_info / cumulative_info[-1]
    nAll = len(s)

    # if nPOD = -1, use cumulative tolerance
    # if nPOD = -2 use all coefficients (or set nPOD = nTime)
    # if nPOD > 0 use nPOD coefficients as defined by the user

    if nPOD == -1:
        # SVD truncation - percentage of information captured or number
        nPOD = sum(cumulative_info <= cumulative_tol)  # tolerance
    elif nPOD == -2:
        nPOD = nAll

    R = u[:, :nPOD]

    coeffs = []

    for iGrid in range(len(snapshots)):
        snapshots_per_grid = \
            out[:, iGrid*snapshots[0].shape[-1]:(iGrid+1) *
                snapshots[0].shape[-1]]

        coeffs.append(np.dot(R.T, snapshots_per_grid))

    s = s[:nPOD]

    return coeffs, R, s


def reconstruct_pod(coeffs, R):
    """
    Reconstruct grid from POD coefficients and transormation matrix R.

    Args:
        coeffs (np.array): POD coefficients
        R (np.array): Transformation matrix R

    Returns:
        np.array: Reconstructed grid
    """

    return R @ coeffs


class mse_weighted:
    """
    Custom weighted mean squared error loss
    """
    def __init__(self) -> None:
        """
        Constructor, name is required for TensorFlow custom losses. Since
        we only know weights after compiling the model needs to be attributes
        """
        self.weights = None
        self.__name__ = "mse_weighted"

    def __call__(self, y_true, y_pred):
        """
        Tensorflow loss needs to be callable.

        Args:
            y_true (np.array or tf.tensor): True values
            y_pred (np.array or tf.tensor): Predicted values

        Returns:
            float: Weighted MSE loss
        """

        return K.mean(K.square(y_pred*self.weights - y_true*self.weights),
                      axis=-1)


class mse_PI:
    """
    Mean squared error loss class.
    """
    def __init__(self, dx=None, dy=None):
        self.dx = dx
        self.dy = dx
        self.__name__ = "mse_PI"

    def __call__(self, y_true, y_pred):
        """
        Call the class, calculate the physics informed MSE loss

        Args:
            y_true (np.array): True values
            y_pred (np.array): Predictions by model

        Raises:
            ValueError: Raises error if intervals dx and dy are not set

        Returns:
            float: Physics informed loss value
        """
        if self.dx is None or self.dy is None:
            raise ValueError("First set dx and dy")

        # cty is the value of the continuity equation
        cty = 0

        # keep a count such that we can average later
        count = 0

        for k in range(y_pred.shape[0]):
            print(k)
            # K is the grid in the batch
            for i in range(1, y_pred.shape[1]-1):
                # index in x direction
                for j in range(1, y_pred.shape[2]-1):
                    # index in y direction
                    cty += (y_pred[k, i+1, j, 0] - y_pred[k, i-1, j, 0]) / \
                             (2*self.dx) + \
                           (y_pred[k, i, j+1, 1] - y_pred[k, i, j-1, 1]) / \
                             (2*self.dy)
                    count += 1

        cty = cty/count

        return K.mean(mse(y_true, y_pred)) + abs(cty)


def print_losses(d_loss, g_loss, epoch, d_loss_val=None, g_loss_val=None):
    """
    Convenience function to print a set of losses. Can be used by adversarial
    type of networks

    Args:
        d_loss (float): Discriminator loss value
        g_loss (float): Generator loss value
        epoch (int): Current epoch
        d_loss_val (float, optional): Validation discriminator loss value. 
                                      Defaults to None.
        g_loss_val (float, optional): Validation generator loss value.
                                      Defaults to None.
    """
    print("%d: [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" %
          (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

    if d_loss_val is not None and g_loss_val is not None:
        print("%d val: [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" %
              (epoch, d_loss_val[0], 100*d_loss_val[1], g_loss_val[0],
               g_loss_val[1]))


def plot_losses(d_loss, g_loss, liveloss, d_loss_val=None,
                g_loss_val=None):
    """
    Convenience function to plot a set of losses. Can be used by adversarial
    type of networks

    Args:
        d_loss (float): Discriminator loss value
        g_loss (float): Generator loss value
        livaloss (object): livelossplot class instance
        d_loss_val (float, optional): Validation discriminator loss value.
                                      Defaults to None.
        g_loss_val (float, optional): Validation generator loss value.
                                      Defaults to None.
    """
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
