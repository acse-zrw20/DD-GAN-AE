"""

Tests for DD-GAN-AE repository. Please execute from root of repository

"""

from pytest import fixture
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
from ddganAE.utils import calc_pod
from ddganAE.preprocessing import convert_2d
from ddganAE.models import CAE
from ddganAE.architectures import *
from sklearn.model_selection import train_test_split

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

"""
def test_cae(snapshots):

    x_train, x_val = train_test_split(snapshots, test_size=0.1)

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    optimizer = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.98, beta_2=0.9)
"""