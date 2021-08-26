"""
Module that wraps some legacy code to get a set of pod coefficients for
subdomains (domain-decomposed) from a domain decomposed flow past cylinder
problem.

Note this code is meant to be a wrapper for legacy code that is intended to
not be used used very often or in a critical/production setting. Therefore
sustainability  may be lacking.
"""

import vtktools
import numpy as np
from utils import get_grid_end_points
import argparse
import sys

# Backwards compatibility with python 2
if sys.version_info[0] < 3:
    import u2r  # noqa
else:
    import u2rpy3  # noqa
    u2r = u2rpy3

__author__ = " Claire Heaney, Zef Wolffs"
__credits__ = ["Jon Atli Tomasson"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


def get_pod_coeffs(data_dir='./submodules/DD-GAN/data/FPC_Re3900_2D_CG_old/',
                   data_file_base='fpc_2D_Re3900_CG_',
                   out_dir='./../../data/processed/', nTime=1400,
                   offset=20, field_names=['Velocity'], nGrids=4, xlength=2.2,
                   ylength=0.41, nloc=3, nScalar=2, nDim=2):
    """
    Function that wraps some legacy code to interpolated data from  an
    unstructured mesh to a structured mesh and calculate POD coefficients from
    output.

    Args:
        data_dir (str, optional): Input data folder.
            Defaults to `./submodules/DD-GAN/data/FPC_Re3900_2D_CG_old/`.
        data_file_base (str, optional): Base filename, timesteps will be
            appended. Defaults to `fpc_`.
        out_dir (str, optional): Output data folder. Defaults to
            `./../../data/processed/`.
        nTime (int, optional): Number of timesteps to include in snapshots
            matrix. Defaults to 1400.
        offset (int, optional): At which time level to start taking the
            snapshots. Defaults to 20.
        field_names (list, optional): Names of fields to include from vtu
            data file. Defaults to [`Velocity`].
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
    """

    nFields = len(field_names)

    if nGrids == 4:
        nx = 55
        ny = 42
        nz = 1  # nz = 1 for 2D problems
    elif nGrids == 1:
        nx = 221
        ny = 42
        nz = 1  # nz = 1 for 2D problems
    else:
        print('nx, ny, nz not known for ', nGrids, 'grids')

    grid_origin = [0.0, 0.0]
    grid_width = [xlength/nGrids, 0.0]
    # ddx = np.array((0.01,0.01))
    ddx = np.array((xlength/(nGrids*(nx-1)), ylength/(ny-1)))
    print('ddx', ddx)

    # get a vtu file (any will do as the mesh is not adapted)
    filename = data_dir + data_file_base + '0.vtu'
    representative_vtu = vtktools.vtu(filename)
    coordinates = representative_vtu.GetLocations()

    nNodes = coordinates.shape[0]  # vtu_data.ugrid.GetNumberOfPoints()
    nEl = representative_vtu.ugrid.GetNumberOfCells()
    print('nEl', nEl, type(nEl), 'nNodes', nNodes)
    # nNodes = 3571
    # nEl = 6850

    x_all = np.transpose(coordinates[:, 0:nDim])  # coords n,3  x_all 2,n

    # get global node numbers
    x_ndgln = np.zeros((nEl*nloc), dtype=int)
    for iEl in range(nEl):
        n = representative_vtu.GetCellPoints(iEl) + 1
        x_ndgln[iEl*nloc:(iEl+1)*nloc] = n

    # -------------------------------------------------------------------------------------------------
    # find node duplications when superposing results
    my_field = representative_vtu.GetField(field_names[0])[:, 0]
    my_field = 1
    nScalar_test = 1
    # for one timestep
    # for one field
    value_mesh = np.zeros((nScalar_test, nNodes, 1))  # nTime=1
    value_mesh[:, :, 0] = np.transpose(my_field)
    superposed_grids = np.zeros((nNodes))
    for iGrid in range(nGrids):
        block_x_start = get_grid_end_points(grid_origin, grid_width, iGrid)

        zeros_on_mesh = 0
        value_grid = u2r.simple_interpolate_from_mesh_to_grid(value_mesh,
                                                              x_all, x_ndgln,
                                                              ddx,
                                                              block_x_start,
                                                              nx, ny, nz,
                                                              zeros_on_mesh,
                                                              nEl, nloc,
                                                              nNodes,
                                                              nScalar_test,
                                                              nDim, 1)

        zeros_on_grid = 1
        value_back_on_mesh = u2r.interpolate_from_grid_to_mesh(value_grid,
                                                               block_x_start,
                                                               ddx, x_all,
                                                               zeros_on_grid,
                                                               nScalar_test,
                                                               nx, ny, nz,
                                                               nNodes, nDim, 1)

        superposed_grids = superposed_grids + \
            np.rint(np.squeeze(value_back_on_mesh))

    superposed_grids = np.array(superposed_grids, dtype='int')
    duplicated_nodal_values = []
    for iNode in range(nNodes):
        if superposed_grids[iNode] == 0:
            # this is bad news - the node hasn't appeared in any grid
            print('zero:', iNode)
        elif superposed_grids[iNode] == 2:
            print('two:', iNode)
            # the node appears in two grids - deal with this later
            duplicated_nodal_values.append(iNode)
        elif superposed_grids[iNode] != 1:
            # most of the nodes will appear in one grid
            print('unknown:', iNode, superposed_grids[iNode])

    # -------------------------------------------------------------------------------------------------
    # build up the snapshots matrix from solutions on each of the grids
    snapshots_data = []
    for iField in range(nFields):
        # nDoF = nNodes # could be different value per field
        snapshots_data.append(np.zeros((nx*ny*nz*nDim, nGrids*nTime)))

    # value_mesh = np.zeros((nScalar,nNodes)) # no need to initialise -
    # overwritten
    for iTime in range(nTime):

        # print('')
        # print('time level', iTime)

        filename = data_dir + data_file_base + str(offset+iTime) + '.vtu'
        vtu_data = vtktools.vtu(filename)

        for iField in range(nFields):

            my_field = vtu_data.GetField(field_names[iField])[:, 0:nDim]

            for iGrid in range(nGrids):

                block_x_start = get_grid_end_points(grid_origin, grid_width,
                                                    iGrid)
                if iTime == 0:
                    print('block_x_start', block_x_start)

                # value_mesh = np.zeros((nScalar,nNodes,nTime)) # nTime - this
                # must need initialising here
                # value_mesh[:,:,iTime] = np.transpose(my_field)

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

    # ---------------------------------------------------------------------------------------
    # apply POD to the snapshots
    # some POD truncation settings
    cumulative_tol = 0.99
    nPOD = [nTime]  # len(nPOD) = nFields
    nPOD = [-2]
    nPOD = [10]  # 100 50 10

    bases = []
    singular_values = []

    for iField in range(nFields):

        snapshots_matrix = snapshots_data[iField]
        nrows, ncols = snapshots_matrix.shape

        SSmatrix = np.dot(snapshots_matrix.T, snapshots_matrix)

        print('SSmatrix', SSmatrix.shape)
        eigvalues, v = np.linalg.eigh(SSmatrix)
        eigvalues = eigvalues[::-1]

        # get rid of small negative eigenvalues (there shouldn't be any as the
        # eigenvalues of a real, symmetric matrix are non-negative, but
        # sometimes very small negative values do appear)
        eigvalues[eigvalues < 0] = 0
        s_values = np.sqrt(eigvalues)
        # print('s values', s_values[0:20])

        singular_values.append(s_values)

        cumulative_info = np.zeros(len(eigvalues))
        for j in range(len(eigvalues)):
            if j == 0:
                cumulative_info[j] = eigvalues[j]
            else:
                cumulative_info[j] = cumulative_info[j-1] + eigvalues[j]

        cumulative_info = cumulative_info / cumulative_info[-1]
        nAll = len(eigvalues)

    # if nPOD = -1, use cumulative tolerance
    # if nPOD = -2 use all coefficients (or set nPOD = nTime)
    # if nPOD > 0 use nPOD coefficients as defined by the user

        if nPOD[iField] == -1:
            raise NotImplementedError("Option not yet implemented!")
            # SVD truncation - percentage of information captured or number
            cumulative_tol = nirom_options.compression.cumulative_tol[iField] # noqa
            nPOD_iField = sum(cumulative_info <= cumulative_tol)  # tolerance
            nPOD[iField] = nPOD_iField
        elif nPOD[iField] == -2:
            nPOD_iField = nAll
            nPOD[iField] = nPOD_iField
        else:
            nPOD_iField = nPOD[iField]

        print("retaining", nPOD_iField, "basis functions of a possible",
              len(eigvalues))

        # nDim should be nScalar?
        basis_functions = np.zeros((nx*ny*nz*nDim, nPOD_iField))
        for j in reversed(range(nAll-nPOD_iField, nAll)):
            Av = np.dot(snapshots_matrix, v[:, j])
            basis_functions[:, nAll-j-1] = Av/np.linalg.norm(Av)

        bases.append(basis_functions)

    pod_coeffs = []

    for iField in range(nFields):

        basis = bases[iField]

        snapshots_matrix = snapshots_data[iField]
        print('snapshots_matrix', snapshots_matrix.shape)

        for iGrid in range(nGrids):

            # :,iTime*nGrids+iGrid
            # want solutions in time for a particular grid
            snapshots_per_grid = np.zeros((nx*ny*nz*nDim, nTime))
            for iTime in range(nTime):
                # print('taking snapshots from', iTime*nGrids+iGrid )
                snapshots_per_grid[:, iTime] = snapshots_matrix[:,
                                                                iTime*nGrids
                                                                + iGrid]
            print(snapshots_per_grid)
            pod_coeffs.append(np.dot(basis.T, snapshots_per_grid))
            print(basis.shape)

        np.save(out_dir + "/pod_coeffs_field_{}".format(
            field_names[iField]), np.array(pod_coeffs))
        np.save(out_dir + "/pod_basis_field_{}".format(
            field_names[iField]), np.array(basis))
        np.savetxt(out_dir + "/pod_eigenvalues_field_{}".format(
            field_names[iField]), np.array(eigvalues))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Module that wraps some \
legacy code to interpolate data from  an unstructured mesh to a structured \
mesh and calculate POD coefficients from output. \n\n\
Note that output will be saved under the folder specified by out_dir as \
`pod_coeffs_field_<name of field>`. \
The shape of the output matrix will be (num_subgrids, num_pod_coeffs, \
num_time_levels)")
    parser.add_argument('--data_dir', type=str, nargs='?',
                        default="./submodules/DD-GAN/data/\
FPC_Re3900_2D_CG_old/",
                        help='Input data folder')
    parser.add_argument('--data_file_base', type=str, nargs='?',
                        default="fpc_",
                        help='Base filename')
    parser.add_argument('--out_dir', type=str, nargs='?',
                        default="./../../data/processed/",
                        help='Output data folder')
    parser.add_argument('--nTime', type=int, nargs='?', default=2000,
                        help='Number of timesteps to include in snapshots')
    parser.add_argument('--offset', type=int, nargs='?', default=0,
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

    get_pod_coeffs(**arg_dict)
