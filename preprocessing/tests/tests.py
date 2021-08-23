import sys
import os
import numpy as np
import pytest
sys.path.insert(1, './../src/')
sys.path.insert(1, './preprocessing/tests/')
from get_pod_coeffs import get_pod_coeffs # noqa F401
from get_snapshots import get_subgrid_snapshots # noqa F401

"""
Please execute module from within preprocessing/tests folder
"""


@pytest.fixture(autouse=True)
def cleanup():
    # Quick check that the data is available. If not, test makes no sense.
    assert os.path.isfile(
        './../../data/FPC_Re3900_2D_CG_old/fpc_2D_Re3900_CG_0.vtu'), \
            "Data files not available! Make sure they are present"

    """
    assert os.path.isfile(
        './test_data/pod_coeffs_field_velocity.npy'), \
        'Correct result files not available!'

    assert os.path.isfile(
        './test_data/subgrid_snapshots.npy'), \
        'Correct result files not available!'
    """

    yield

    try:
        os.remove('./pod_coeffs_field_Velocity.npy')
    except OSError:
        pass


def test_get_pod_coeffs():
    """
    Regression test with correct results from before
    """
    get_pod_coeffs(out_dir='.')
    coeffs = np.load('pod_coeffs_field_Velocity.npy')
    assert coeffs.shape == (4, 10, 200)

    # Regression test on data calculated earlier
    coeffs_correct = np.load('./test_data/pod_coeffs_field_Velocity.npy')

    assert (coeffs == coeffs_correct).all


def test_get_subgrid_snapshots():
    """
    Regression test with correct results from before
    """
    subgrid_snapshots = get_subgrid_snapshots()

    subgrid_snapshots_corr = np.load('./test_data/subgrid_snapshots.npy')

    assert (subgrid_snapshots == subgrid_snapshots_corr).all()
