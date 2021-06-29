"""

Tests for DD-GAN-AE repository. Please execute from root of repository

"""

from pytest import fixture
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
from ddganAE.utils import calc_pod

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

    snapshots_grids = np.load("./tests/data/test_snapshots.npy")

    return snapshots_grids


def test_POD(snapshots):
    """
    Test that setting a seed works as expected

    Args:
        ddgan (module): ddgan module with all functions
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
