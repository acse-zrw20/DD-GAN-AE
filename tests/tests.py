"""

Tests for DD-GAN-AE repository. Please execute from root of repository

"""

from pytest import fixture
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
from ddganAE.utils import calc_pod, mse_weighted
from ddganAE.preprocessing import convert_2d

__author__ = "Zef Wolffs"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


@fixture(scope='module')
def snapshots():
    """
    Load in snapshots
    """

    snapshots_grids = np.load("./tests/data/test_snapshots_fpc.npy")

    return snapshots_grids


def test_POD(snapshots):
    """
    Test that POD works as expected

    Args:
        snapshots (np.array): snapshots data
    """

    # Data normalization
    layer = preprocessing.Normalization(axis=None)
    layer.adapt(snapshots)

    snapshots = layer(snapshots).numpy()

    # Do POD
    coeffs, R, s = calc_pod(snapshots, nPOD=50)

    # Calculate MSE and assert it is small
    mean = 0
    for j in range(4):
        recon = R @ coeffs[j]
        for i in range(200):
            mean += tf.keras.losses.MSE(recon[:, i],
                                        snapshots[j, :, i]).numpy()/800

    assert mean < 1e-3


def test_convert_2D(snapshots):
    """
    Test that the preprocessing utility to convert to 2D works as expected

    Args:
        snapshots (np.array): snapshots data
    """

    # Do the conversion to 2D and create numpy array
    input_shape = (55, 42, 2)
    snapshots = convert_2d(snapshots, input_shape, 200)
    snapshots = np.array(snapshots).reshape(800, *input_shape)

    assert snapshots.shape == (800, 55, 42, 2)

    # Load in the correct snapshots
    snapshots_corr = np.load(
        "./tests/data/test_snapshots_fpc_2D_converted.npy")

    assert np.allclose(snapshots_corr, snapshots)


def test_mse_weighted():
    """
    Test functionality of weighted MSE class
    """
    loss = mse_weighted()
    loss.weights = np.array([1, 2])

    assert 3 == loss(np.array([1, 1]), np.array([0, 0]))
