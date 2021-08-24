"""
Module that wraps some legacy code to get a set of snapshots and domain
decompose given flow past cylinder problem.

Code is not very general and likely only works for exact flow past cylinder
used in this project. Note this code is meant to be a wrapper for legacy code
that is intended to not be used used very often or in a critical/production
setting. Therefore sustainability may be lacking.
"""

import vtktools
import numpy as np
from utils import get_grid_end_points
import argparse
import sys

if sys.version_info[0] < 3:
    import u2r # noqa
else:
    import u2rpy3 # noqa
    u2r = u2rpy3

__author__ = " Claire Heaney, Zef Wolffs"
__credits__ = ["Jon Atli Tomasson"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


def get_subgrid_snapshots(
    data_dir='./submodules/DD-GAN/data/FPC_Re3900_2D_CG_old/',
    data_file_base='fpc_2D_Re3900_CG_',
    out_dir='.', nTime=200,
    offset=500, field_names=['Velocity'], nGrids=4,
    xlength=2.2, ylength=0.41, nloc=3, nScalar=2, nDim=2
        ):

    """
    Function that wraps some legacy code to interpolate data from an
    unstructured mesh to a structured mesh and does domain decomposition.


    Args:
        data_dir (str, optional): Input data folder.
            Defaults to './../../data/FPC_Re3900_2D_CG_old/'.
        data_file_base (str, optional): Base filename, timesteps will be
            appended. Defaults to 'fpc_2D_Re3900_CG_'.
        out_dir (str, optional): Output data folder. Defaults to
            './../../data/processed/'.
        nTime (int, optional): Number of timesteps to include in snapshots
            matrix. Defaults to 200.
        offset (int, optional): At which time level to start taking the
            snapshots. Defaults to 500.
        field_names (list, optional): Names of fields to include from vtu
            data file. Defaults to ['Velocity'].
        nGrids (int, optional): Number of grids of decomposed domain,
            choose 1 or 4. Defaults to 4.
        xlength (float, optional): Length of interpolated domain in x.
            Defaults to 2.2.
        ylength (float, optional): Length of interpolated domain in y.
            Defaults to 0.41.
        nloc (int, optional): Number of local nodes, ie three nodes per
            element (in 2D). Defaults to 3.
        nScalar (int, optional): Dimension of fields. Defaults to 2.
        nDim (int, optional): Dimension of problem. Defaults to 2.

    Returns:
        list: List of arrays that form the snapshots of the subdomains
    """

    # nDim
    nFields = len(field_names)

    # Currently leave this hardcoded
    if nGrids == 4:
        nx = 55
        ny = 42
        nz = 1
    elif nGrids == 1:
        nx = 221
        ny = 42
        nz = 1
    else:
        raise ValueError('nx, ny, nz not known for ', nGrids, 'grids')

    grid_origin = [0.0, 0.0]
    grid_width = [xlength/nGrids, 0.0]

    ddx = np.array((xlength/(nGrids*(nx-1)), ylength/(ny-1)))

    # get a vtu file (any will do as the mesh is not adapted)
    filename = data_dir + data_file_base + '0.vtu'
    representative_vtu = vtktools.vtu(filename)
    coordinates = representative_vtu.GetLocations()

    nNodes = coordinates.shape[0]  # vtu_data.ugrid.GetNumberOfPoints()
    nEl = representative_vtu.ugrid.GetNumberOfCells()

    x_all = np.transpose(coordinates[:, 0:nDim])

    # get global node numbers
    x_ndgln = np.zeros((nEl*nloc), dtype=int)
    for iEl in range(nEl):
        n = representative_vtu.GetCellPoints(iEl) + 1
        x_ndgln[iEl*nloc:(iEl+1)*nloc] = n

    # -------------------------------------------------------------------------------------------------
    # build up the snapshots matrix from solutions on each of the grids
    snapshots_data = []
    for iField in range(nFields):
        snapshots_data.append(np.zeros((nx*ny*nz*nDim, nGrids*nTime)))

    # value_mesh = np.zeros((nScalar,nNodes)) # no need to initialise -
    # overwritten
    for iTime in range(nTime):

        filename = data_dir + data_file_base + str(offset+iTime) + '.vtu'
        vtu_data = vtktools.vtu(filename)

        for iField in range(nFields):

            my_field = vtu_data.GetField(field_names[iField])[:, 0:nDim]

            for iGrid in range(nGrids):

                block_x_start = get_grid_end_points(grid_origin, grid_width,
                                                    iGrid)

                value_mesh = np.transpose(my_field)  # size nScalar,nNodes

                # interpolate field onto structured mesh
                # feed in one result at t time (no need to store in value_mesh
                # in this case)

                # 0 extrapolate solution (for the cylinder in fpc); 1 gives
                # zeros for nodes outside mesh
                zeros_beyond_mesh = 0

                value_grid = \
                    u2r.simple_interpolate_from_mesh_to_grid(value_mesh, x_all,
                                                             x_ndgln, ddx,
                                                             block_x_start, nx,
                                                             ny, nz,
                                                             zeros_beyond_mesh,
                                                             nEl, nloc, nNodes,
                                                             nScalar, nDim, 1)

                snapshots_data[iField][:, iTime*nGrids+iGrid] = \
                    value_grid.reshape(-1)

    subgrid_snapshots = []
    for iField in range(nFields):

        snapshots_matrix = snapshots_data[iField]

        for iGrid in range(nGrids):

            # :,iTime*nGrids+iGrid
            # want solutions in time for a particular grid
            snapshots_per_grid = np.zeros((nx*ny*nz*nDim, nTime))
            for iTime in range(nTime):
                snapshots_per_grid[:, iTime] = \
                    snapshots_matrix[:, iTime*nGrids+iGrid]

            subgrid_snapshots.append(snapshots_per_grid)

    np.save(out_dir + "/snaphsots_field_{}".format(field_names[iField]),
            np.array(subgrid_snapshots))

    return subgrid_snapshots


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Module that wraps some \
legacy code to interpolate data from  an unstructured mesh to a structured \
mesh and calculate subgrid snapshots from output.")
    parser.add_argument('--data_dir', type=str, nargs='?',
                        default="./../../data/FPC_Re3900_2D_CG_new/",
                        help='Input data folder')
    parser.add_argument('--data_file_base', type=str, nargs='?',
                        default="fpc_",
                        help='Base filename')
    parser.add_argument('--out_dir', type=str, nargs='?',
                        default="./../../data/processed/",
                        help='Output data folder')
    parser.add_argument('--nTime', type=int, nargs='?', default=200,
                        help='Number of timesteps to include in snapshots')
    parser.add_argument('--offset', type=int, nargs='?', default=500,
                        help='At which time level to start taking snapshots')
    parser.add_argument('--field_names', type=list, nargs='?',
                        default=['Velocity'],
                        help='Names of fields to include from vtu data file')
    parser.add_argument('--nGrids', type=int, nargs='?', default=4,
                        help='Number of grids of decomposed domain, 1 or 4')
    parser.add_argument('--xlength', type=float, nargs='?', default=2.2,
                        help='Length of interpolated domain in x')
    parser.add_argument('--ylength', type=float, nargs='?', default=0.41,
                        help='Length of interpolated domain in y')
    parser.add_argument('--nloc', type=int, nargs='?', default=3,
                        help='Number of local nodes')
    parser.add_argument('--nScalar', type=int, nargs='?', default=2,
                        help='Dimension of fields')
    parser.add_argument('--nDim', type=int, nargs='?', default=2,
                        help='Dimension of problem.')
    args = parser.parse_args()

    arg_dict = vars(args)

    get_subgrid_snapshots(**arg_dict)
