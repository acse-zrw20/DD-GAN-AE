"""
Module wraps some legacy code to construct a series of vtu files with 3D
CFD data on unstructured mesh from structured mesh in numpy format.

Code is not very general and likely only works for exact slug flow data
used in this project. Note this code is meant to be a wrapper for legacy code
that is intended to not be used used very often or in a critical/production
setting. Therefore sustainability may be lacking.
"""

import numpy as np
import argparse
import sys
import os
from utils import get_grid_end_points

sys.path.append("/usr/lib/python2.7/dist-packages/")
import vtktools  # noqa

if sys.version_info[0] < 3:
    import u2r # noqa
else:
    import u2rpy3 # noqa
    u2r = u2rpy3

__author__ = "Claire Heaney, Zef Wolffs"
__credits__ = ["Jon Atli Tomasson"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


def create_vtu_file_v_and_a(
    path,
    nNodes,
    v_value_mesh_twice_interp,
    a_value_mesh_twice_interp,
    filename,
    orig_vel,
    orig_alpha,
    iTime,
    nDim=3,
):
    """
    Creates a vtu file with velocity and alpha components reconstructed

    Args:
        path (string): Location of vtu file
        nNodes (int): Number of nodes in vtu file
        v_value_mesh_twice_interp (np.array): velocity mesh
        a_value_mesh_twice_interp (np.array): alpha mesh
        filename (string): name of output file
        orig_vel (np.array): Original velocity mesh
        orig_alpha (np.array): Original Alpha mesh
        iTime (int): Time
        nDim (int, optional): Number of dimensions. Defaults to 2.

    Returns:
        [type]: [description]
    """
    nDim += 1
    velocity_field = np.zeros((nNodes, 3))
    velocity_field[:, 0:nDim] = np.transpose(
        v_value_mesh_twice_interp[0:nDim, :]
    )

    # streamwise component only
    v_difference = np.zeros((nNodes, 3))

    v_difference[:, 0:nDim] = (
        np.transpose(v_value_mesh_twice_interp[0:nDim, :]) - orig_vel
    )

    # streamwise component only
    v_difference = v_difference / np.max(velocity_field)

    # ALPHA #
    alpha_field = np.zeros((nNodes, 3))
    alpha_field[:, 0:nDim] = np.transpose(a_value_mesh_twice_interp[0:nDim, :])

    # streamwise component only
    a_difference = np.zeros((nNodes, 3))
    a_difference[:, 0:nDim] = (
        np.transpose(a_value_mesh_twice_interp[0:nDim, :]) - orig_alpha
    )

    # streamwise component only
    a_difference = a_difference / np.max(alpha_field)

    clean_vtk = get_clean_vtk_file(filename)
    new_vtu = vtktools.vtu()
    new_vtu.ugrid.DeepCopy(clean_vtk.ugrid)
    new_vtu.filename = path + "recon_" + str(iTime) + ".vtu"
    new_vtu.AddField("Velocity", velocity_field)
    new_vtu.AddField("V_Original", orig_vel)
    new_vtu.AddField("Velocity_diff", v_difference)

    new_vtu.AddField("Alpha", alpha_field)
    new_vtu.AddField("A_Original", orig_alpha)
    new_vtu.AddField("Alpha_diff", a_difference)
    new_vtu.Write()
    return 0


def get_clean_vtk_file(filename):
    """
    Removes fields and arrays from a vtk file,
    leaving the coordinates/connectivity information.
    """
    vtu_data = vtktools.vtu(filename)
    clean_vtu = vtktools.vtu()
    clean_vtu.ugrid.DeepCopy(vtu_data.ugrid)
    fieldNames = clean_vtu.GetFieldNames()
    # remove all fields and arrays from this vtu
    for field in fieldNames:
        clean_vtu.RemoveField(field)
        fieldNames = clean_vtu.GetFieldNames()
        vtkdata = clean_vtu.ugrid.GetCellData()
        arrayNames = [
            vtkdata.GetArrayName(i) for i in range(vtkdata.GetNumberOfArrays())
        ]
    for array in arrayNames:
        vtkdata.RemoveArray(array)
    return clean_vtu


def reconstruct_3D(
    out_file_base="slug_255_exp_projected_",
    nGrids=10,
    offset=0,
    nTime=2,
    input_array="cae_reconstruction_sf.npy",
):
    """
    Go from numpy array grid to vtu file mesh

    Args:
        out_file_base (str, optional): Example file base, will also be used for
                                   original velocity and alpha. Defaults to
                                   "slug_255_exp_projected_".
        nGrids (int, optional): Number of grids. Defaults to 10.
        offset (int, optional): Time offset. Defaults to 0.
        nTime (int, optional): Number of timesteps. Defaults to 2.
        input_array (str, optional): Input array to convert to vtu,
                                     expects shape to be (ngrids*ntime, nx,
                                     ny, nz, nscalar_vel+nscalar_alpha).
                                     Defaults to "dataset.npy".
    """

    filename = out_file_base + "0" + ".vtu"
    vtu_file = vtktools.vtu(filename)
    coordinates = vtu_file.GetLocations()

    nNodes = coordinates.shape[0]  # vtu_data.ugrid.GetNumberOfPoints()
    nEl = vtu_file.ugrid.GetNumberOfCells()
    nScalar = 3  # dimension of fields
    nScalar_alpha = 1
    nDim = 3  # dimension of problem (no need to interpolate in dim no 3)
    nloc = 4  # number of local nodes, ie four nodes per element (in 3D)

    # get global node numbers
    x_ndgln = np.zeros((nEl * nloc), dtype=int)
    for iEl in range(nEl):
        n = vtu_file.GetCellPoints(iEl) + 1
        x_ndgln[iEl * nloc: (iEl + 1) * nloc] = n

    nx = 60
    ny = 20
    nz = 20

    xlength = 10  # length of full grid
    ylength = 0.078
    zlength = 0.078

    x_all = np.transpose(coordinates[:, 0:nDim])

    ddx = np.array(
        (
            float(xlength) / (nGrids * (nx - 1)),
            ylength / (ny - 1),
            zlength / (nz - 1),
        )
    )

    grid_origin = [0, -0.039, -0.039]
    grid_width = [xlength / nGrids, 0.0, 0.0]

    # First let's find the duplicated nodal values
    my_field = vtu_file.GetField("phase1::Velocity")[:, 0]
    my_field = 1
    nScalar_test = 1

    value_mesh = np.zeros((nScalar_test, nNodes, 1))  # nTime=1
    value_mesh[:, :, 0] = np.transpose(my_field)
    superposed_grids = np.zeros((nNodes))

    for iGrid in range(nGrids):
        block_x_start = get_grid_end_points(grid_origin, grid_width, iGrid)

        zeros_on_mesh = 0
        value_grid = u2r.simple_interpolate_from_mesh_to_grid(
            value_mesh,
            x_all,
            x_ndgln,
            ddx,
            block_x_start,
            nx,
            ny,
            nz,
            zeros_on_mesh,
            nEl,
            nloc,
            nNodes,
            nScalar_test,
            nDim,
            1,
        )

        zeros_on_grid = 1
        value_back_on_mesh = u2r.interpolate_from_grid_to_mesh(
            value_grid,
            block_x_start,
            ddx,
            x_all,
            zeros_on_grid,
            nScalar_test,
            nx,
            ny,
            nz,
            nNodes,
            nDim,
            1,
        )

        superposed_grids = superposed_grids + np.rint(
            np.squeeze(value_back_on_mesh)
        )

    superposed_grids = np.array(superposed_grids, dtype="int")
    duplicated_nodal_values = []
    for iNode in range(nNodes):
        if superposed_grids[iNode] == 0:
            # this is bad news - the node hasn't appeared in any grid
            print("zero:", iNode)
        elif superposed_grids[iNode] == 2:
            print("two:", iNode)
            # the node appears in two grids - deal with this later
            duplicated_nodal_values.append(iNode)
        elif superposed_grids[iNode] != 1:
            # most of the nodes will appear in one grid
            print("unknown:", iNode, superposed_grids[iNode])

    # Now we can do the actual reconstruction
    reconstruction_on_mesh = np.zeros((nScalar * nTime, nNodes))
    grids = np.load(input_array)

    grids_split_domains = np.zeros(
        (
            nGrids,
            nTime,
            grids.shape[1],
            grids.shape[2],
            grids.shape[3],
            grids.shape[4],
        )
    )

    for i in range(nTime):
        grids_split_domains[:, i, :, :, :, :] = grids[
            (i) * (nGrids): (i + 1) * nGrids, :, :, :
        ]

    grids = np.swapaxes(grids_split_domains, 1, 5)

    velocity_reconstructed = grids[:, :3, :, :, :, :]
    alpha_reconstructed = grids[:, 3:4, :, :, :, :]

    # We first do velocities
    for iGrid in range(nGrids):

        reconstruction_grid = velocity_reconstructed[iGrid, :, :, :, :, :]
        # reconstruction_grid here has the shape of (nScalar, nx, ny, nz,
        #                                            nTime)

        block_x_start = get_grid_end_points(grid_origin, grid_width, iGrid)

        for iTime in range(nTime):

            zeros_beyond_grid = 1  # 0 extrapolate solution; 1 gives zeros
            reconstruction_on_mesh_from_one_grid = (
                u2r.interpolate_from_grid_to_mesh(
                    reconstruction_grid[:, :, :, :, iTime],
                    block_x_start,
                    ddx,
                    x_all,
                    zeros_beyond_grid,
                    nScalar,
                    nx,
                    ny,
                    nz,
                    nNodes,
                    nDim,
                    1,
                )
            )

            reconstruction_on_mesh[
                nScalar * iTime: nScalar * (iTime + 1), :
            ] = reconstruction_on_mesh[
                nScalar * iTime: nScalar * (iTime + 1), :
            ] + np.squeeze(
                reconstruction_on_mesh_from_one_grid
            )

    reconstruction_on_mesh[:, duplicated_nodal_values] = (
        0.5 * reconstruction_on_mesh[:, duplicated_nodal_values]
    )

    velocity_reconstruction_on_mesh = reconstruction_on_mesh

    # for ifield in range(nFields):
    #    nDoF = nNodes # could be different value per field
    #    original_data.append(np.zeros((nNodes, nDim*nTime)))
    original_velocity = np.zeros((nNodes, nDim * nTime))
    for iTime in range(nTime):
        filename = out_file_base + str(offset + iTime) + ".vtu"
        vtu_data = vtktools.vtu(filename)

        my_field = vtu_data.GetField("phase1::Velocity")[:, 0:nDim]
        original_velocity[:, iTime * nDim: (iTime + 1) * nDim] = my_field

    reconstruction_on_mesh = np.zeros((nScalar_alpha * nTime, nNodes))

    for iGrid in range(nGrids):

        reconstruction_grid = alpha_reconstructed[iGrid, :, :, :, :, :]
        # reconstruction_grid here has the shape of (nScalar, nx, ny, nz,
        #                                            nTime)

        block_x_start = get_grid_end_points(grid_origin, grid_width, iGrid)

        print(block_x_start)

        for iTime in range(nTime):

            zeros_beyond_grid = 1  # 0 extrapolate solution; 1 gives zeros
            reconstruction_on_mesh_from_one_grid = (
                u2r.interpolate_from_grid_to_mesh(
                    reconstruction_grid[:, :, :, :, iTime],
                    block_x_start,
                    ddx,
                    x_all,
                    zeros_beyond_grid,
                    nScalar_alpha,
                    nx,
                    ny,
                    nz,
                    nNodes,
                    nDim,
                    1,
                )
            )

            reconstruction_on_mesh[
                nScalar_alpha * iTime: nScalar_alpha * (iTime + 1), :
            ] = reconstruction_on_mesh[
                nScalar_alpha * iTime: nScalar_alpha * (iTime + 1), :
            ] + np.squeeze(
                reconstruction_on_mesh_from_one_grid
            )

    reconstruction_on_mesh[:, duplicated_nodal_values] = (
        0.5 * reconstruction_on_mesh[:, duplicated_nodal_values]
    )

    alpha_reconstruction_on_mesh = reconstruction_on_mesh

    # for ifield in range(nFields):
    #    nDoF = nNodes # could be different value per field
    #    original_data.append(np.zeros((nNodes, nDim*nTime)))
    original_alpha = np.zeros((nNodes, nDim * nTime))
    for iTime in range(nTime):
        filename = out_file_base + str(offset + iTime) + ".vtu"
        vtu_data = vtktools.vtu(filename)

        my_field = vtu_data.GetField(
            "Component1::ComponentMassFractionPhase1"
        )[:, 0:nDim]
        original_alpha[:, iTime * nDim: (iTime + 1) * nDim] = my_field

    # make diretory for results
    path_to_reconstructed_results = "reconstructed_results/"
    if not os.path.isdir(path_to_reconstructed_results):
        os.mkdir(path_to_reconstructed_results)

    template_vtu = out_file_base + "0.vtu"
    for iTime in range(nTime):
        create_vtu_file_v_and_a(
            path_to_reconstructed_results,
            nNodes,
            velocity_reconstruction_on_mesh[
                iTime * nScalar: (iTime + 1) * nScalar, :
            ],
            alpha_reconstruction_on_mesh[
                iTime * nScalar_alpha: (iTime + 1) * nScalar_alpha, :
            ],
            template_vtu,
            original_velocity[:, iTime * nDim: (iTime + 1) * nDim],
            original_alpha[:, iTime * nDim: (iTime + 1) * nDim],
            iTime,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module that wraps some \
legacy code to interpolate data from  a structured mesh to an unstructured \
mesh and calculate vtu files from output for 3D slug flow dataset."
    )
    parser.add_argument(
        "--out_file_base",
        type=str,
        nargs="?",
        default="slug_255_exp_projected_",
        help=" Example file base, will also be used for\
 original velocity and alpha",
    )
    parser.add_argument(
        "--offset",
        type=int,
        nargs="?",
        default=0,
        help="vtu file to start from",
    )
    parser.add_argument(
        "--nTime", type=int, nargs="?", default=2, help="number of timesteps"
    )
    parser.add_argument(
        "--input_array",
        type=str,
        nargs="?",
        default="cae_reconstruction_sf.npy",
        help="Input array to convert to vtu,\
 expects shape to be (ngrids*ntime, nx,\
 ny, nz, nscalar_vel+nscalar_alpha)",
    )
    args = parser.parse_args()

    arg_dict = vars(args)

    reconstruct_3D(**arg_dict)
