"""
Module wraps some legacy code to construct a series of vtu files with 2D
CFD data on unstructured mesh from structured mesh in numpy format.

Code is not very general and likely only works for exact flow past cylinder
dataset used in this project. Note this code is meant to be a wrapper for
legacy code that is intended to not be used used very often or in a
critical/production setting. Therefore sustainability may be lacking.
"""

import vtktools
import numpy as np
from utils import get_grid_end_points
import os
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


def create_vtu_file(
    path, nNodes, value_mesh_twice_interp, filename, orig_vel, iTime, nDim=2
):
    velocity_field = np.zeros((nNodes, 3))
    velocity_field[:, 0:nDim] = np.transpose(
        value_mesh_twice_interp[0:nDim, :]
    )

    # streamwise component only
    difference = np.zeros((nNodes, 3))
    difference[:, 0:nDim] = (
        np.transpose(value_mesh_twice_interp[0:nDim, :]) - orig_vel
    )

    # streamwise component only
    difference = difference / np.max(velocity_field)

    clean_vtk = get_clean_vtk_file(filename)
    new_vtu = vtktools.vtu()
    new_vtu.ugrid.DeepCopy(clean_vtk.ugrid)
    new_vtu.filename = path + "recon_" + str(iTime) + ".vtu"
    new_vtu.AddField("Velocity", velocity_field)
    new_vtu.AddField("Original", orig_vel)
    new_vtu.AddField("Velocity_diff", difference)
    new_vtu.Write()
    return


def reconstruct(
    snapshot_data_location="./../../data/FPC_Re3900_2D_CG_new/",
    snapshot_file_base="fpc_",
    reconstructed_file="reconstruction_test.npy",  # POD coefficients
    nGrids=4,
    xlength=2.2,
    ylength=0.41,
    nTime=300,
    field_names=["Velocity"],
    offset=0
):
    """
    Requires data in format (ngrids, nscalar, nx, ny, ntime)

    Args:
        snapshot_data_location (str, optional): location of sample vtu file.
                                                Defaults to
                                                "./../../data/FPC_Re3900_2D_CG_new/".
        snapshot_file_base (str, optional): file base of sample vtu file.
                                            Defaults to "fpc_".
        reconstructed_file (str, optional): reconstruction data file. Defaults
                                            to "reconstruction_test.npy".
        xlength (float, optional): length in x direction. Defaults to 2.2.
        ylength (float, optional): length in y direction. Defaults to 0.41.
        nTime (int, optional): number of timesteps. Defaults to 300.
        field_names (list, optional): names of fields in vtu file. Defaults to
                                      ["Velocity"].
        offset (int, optional): starting timestep. Defaults to 0.
    """

    nFields = len(field_names)

    # get a vtu file (any will do as the mesh is not adapted)
    filename = snapshot_data_location + snapshot_file_base + "0.vtu"
    representative_vtu = vtktools.vtu(filename)
    coordinates = representative_vtu.GetLocations()

    nNodes = coordinates.shape[0]  # vtu_data.ugrid.GetNumberOfPoints()
    nEl = representative_vtu.ugrid.GetNumberOfCells()
    nScalar = 2  # dimension of fields
    nDim = 2  # dimension of problem (no need to interpolate in dim no 3)
    nloc = 3  # number of local nodes, ie three nodes per element (in 2D)

    # get global node numbers
    x_ndgln = np.zeros((nEl * nloc), dtype=int)
    for iEl in range(nEl):
        n = representative_vtu.GetCellPoints(iEl) + 1
        x_ndgln[iEl * nloc: (iEl + 1) * nloc] = n

    # set grid size
    if nGrids == 4:
        nx = 55
        ny = 42
        nz = 1  # nz = 1 for 2D problems
    elif nGrids == 1:
        nx = 221
        ny = 42
        nz = 1  # nz = 1 for 2D problems
    else:
        print("nx, ny, nz not known for ", nGrids, "grids")

    x_all = np.transpose(coordinates[:, 0:nDim])

    ddx = np.array((xlength / (nGrids * (nx - 1)), ylength / (ny - 1)))

    grid_origin = [0.0, 0.0]
    grid_width = [xlength / nGrids, 0.0]

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

    reconstruction_on_mesh = np.zeros((nScalar * nTime, nNodes))

    reconstructed = np.load(reconstructed_file)

    for iGrid in range(nGrids):

        reconstruction_grid = reconstructed[iGrid, :, :, :, :]
        # reconstruction_grid here has the shape of (nScalar, nx, ny, nTime)

        block_x_start = get_grid_end_points(grid_origin, grid_width, iGrid)

        for iTime in range(nTime):

            zeros_beyond_grid = 1  # 0 extrapolate solution; 1 gives zeros
            reconstruction_on_mesh_from_one_grid = (
                u2r.interpolate_from_grid_to_mesh(
                    reconstruction_grid[:, :, :, iTime],
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

    # for ifield in range(nFields):
    #    nDoF = nNodes # could be different value per field
    #    original_data.append(np.zeros((nNodes, nDim*nTime)))
    original = np.zeros((nNodes, nDim * nTime))
    for iTime in range(nTime):
        filename = (
            snapshot_data_location
            + snapshot_file_base
            + str(offset + iTime)
            + ".vtu"
        )
        vtu_data = vtktools.vtu(filename)

        for iField in range(nFields):
            my_field = vtu_data.GetField(field_names[iField])[:, 0:nDim]
            original[:, iTime * nDim: (iTime + 1) * nDim] = my_field

    # make diretory for results
    path_to_reconstructed_results = "reconstructed_results/"
    if not os.path.isdir(path_to_reconstructed_results):
        os.mkdir(path_to_reconstructed_results)

    template_vtu = snapshot_data_location + snapshot_file_base + "0.vtu"
    for iTime in range(nTime):
        create_vtu_file(
            path_to_reconstructed_results,
            nNodes,
            reconstruction_on_mesh[iTime * nScalar: (iTime + 1) * nScalar, :],
            template_vtu,
            original[:, iTime * nDim: (iTime + 1) * nDim],
            iTime,
        )


if __name__ == "__main__":
    reconstruct()
