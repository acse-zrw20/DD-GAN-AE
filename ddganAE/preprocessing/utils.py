import numpy as np


def convert_2d(subgrid_snapshots, shape, timesteps):
    """
    Utility to convert list of grids to list of 2d grids

    Args:
        subgrid_snapshots (List): List of subgrids
        shape (Tuple): Shape of 2d grid, e.g. (nFields, nx, ny)
        timesteps (Int): Number of timesteps

    Returns:
        List: List of converted subgrids
    """
    subgrid_snapshots_out = []
    for i, subgrid_snapshot in enumerate(subgrid_snapshots):
        subgrid_snapshot = subgrid_snapshot.reshape((shape[2],
                                                     shape[0],
                                                     shape[1],
                                                     timesteps))
        subgrid_snapshot = np.moveaxis(subgrid_snapshot, 0, 2)
        subgrid_snapshot = np.moveaxis(subgrid_snapshot, 3, 0)
        subgrid_snapshots_out.append(subgrid_snapshot)

    return subgrid_snapshots_out

