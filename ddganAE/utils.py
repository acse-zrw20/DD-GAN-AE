import numpy as np
import keras.backend as K
import tensorflow as tf


def calc_pod(snapshots, nPOD=-2, cumulative_tol=0.99):
    """
    Calculate POD coefficients and basis functions

    Args:
        snapshots (list of ndarrays): List of arrays with subgrid
                                        snapshots. shape:
                                        (n_grids, n_nodes, n_timelevels)

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

    return R @ coeffs


def create_weighted_mse(weights):
    def mse_weighted(y_true, y_pred):
        return K.mean(K.square(y_pred*weights - y_true*weights), axis=-1)

    return mse_weighted


class mse_weighted:

    def __init__(self) -> None:
        self.weights = None
        self.__name__ = "mse_weighted"

    def __call__(self, y_true, y_pred):
        """
        For debugging:

        K.print_tensor(K.mean(K.square(y_pred - y_true),
                              axis=-1))
        K.print_tensor(K.mean(K.square(y_pred*self.weights -
                                       y_true*self.weights),
                              axis=-1))
        K.print_tensor(self.weights)
        """
        return K.mean(K.square(y_pred*self.weights - y_true*self.weights),
                      axis=-1)
