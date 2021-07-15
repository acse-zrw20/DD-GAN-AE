"""

General utilities for package

"""

import numpy as np
import keras.backend as K
from keras.losses import mse
from scipy.sparse.linalg import svds

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
                                        (n_grids, n_nodes*n_scalar, 
                                        n_timelevels)

    Returns:
        list of ndarrays: POD coefficients per subgrid
    """

    # Reshape to have multiple subgrids account for multiple batches
    out = np.zeros((snapshots[0].shape[0],
                    len(snapshots)*snapshots[0].shape[-1]))
    for i, coeff in enumerate(snapshots):
        out[:, i*snapshots[0].shape[-1]:(i+1)*snapshots[0].shape[-1]] = coeff

    snapshots_matrix = out
    nrows, ncols = snapshots_matrix.shape

    if nrows > ncols/4:
        SSmatrix = np.dot(snapshots_matrix.T, snapshots_matrix)
    else:
        SSmatrix = np.dot(snapshots_matrix, snapshots_matrix.T)
        print('WARNING - CHECK HOW THE BASIS FUNCTIONS ARE CALCULATED WITH THIS METHOD')

    print('SSmatrix', SSmatrix.shape)
    eigvalues, v = np.linalg.eigh(SSmatrix)
    eigvalues = eigvalues[::-1]
    # get rid of small negative eigenvalues (there shouldn't be any as the
    # eigenvalues of a real, symmetric
    # matrix are non-negative, but sometimes very small negative values do
    # appear)
    eigvalues[eigvalues < 0] = 0
    s = np.sqrt(eigvalues)
    # print('s values', s_values[0:20]) 

    cumulative_info = np.zeros(len(eigvalues))
    for j in range(len(eigvalues)):
        if j == 0:
            cumulative_info[j] = eigvalues[j]
        else:
            cumulative_info[j] = cumulative_info[j-1] + eigvalues[j]

    cumulative_info = cumulative_info / cumulative_info[-1]
    nAll = len(eigvalues)

    basis_functions = np.zeros((out.shape[0], nPOD))  # nDim should be nScalar?
    for j in reversed(range(nAll-nPOD, nAll)):
        Av = np.dot(snapshots_matrix, v[:, j])
        basis_functions[:, nAll-j-1] = Av/np.linalg.norm(Av)

    R = basis_functions

    coeffs = []

    for iGrid in range(len(snapshots)):
        snapshots_per_grid = \
            out[:, iGrid*snapshots[0].shape[-1]:(iGrid+1) *
                snapshots[0].shape[-1]]

        coeffs.append(np.dot(R.T, snapshots_per_grid))

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
