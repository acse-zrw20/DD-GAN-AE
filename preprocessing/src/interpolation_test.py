import u2r
import numpy as np
import sys, os
sys.path.append('/usr/lib/python2.7/dist-packages/')
import vtk, vtktools

#import matplotlib.pyplot as plt
#import matplotlib.ticker as mticker
#from matplotlib.patches import Circle, PathPatch
#import matplotlib.cm as cm

from collections import OrderedDict

# f2py -c unstruc_mesh_2_regular_grid_new.f90 -m u2r
# help(u2r)
print(u2r.simple_interpolate_from_mesh_to_grid.__doc__)

# test for u2r.simple_interpolate_from_mesh_to_grid

# info from vtu file - has DG velocities
i=100
filename = 'LSBU_' +str(i)+ '.vtu' 
vtu_data =  vtktools.vtu(filename)
tracer = vtu_data.GetField('Tracer')
print('shape tracer', tracer.shape)
coordinates = vtu_data.GetLocations()

nNodes = coordinates.shape[0] # vtu_data.ugrid.GetNumberOfPoints()
print('nNodes', nNodes) 
nEl = vtu_data.ugrid.GetNumberOfCells()
print('nEl', nEl, type(nEl)) # 6850

x_all = np.transpose(coordinates[:,0:3])

# hardwire for lazyness
# only checking one time level and x component of velocity here 
nscalar = 1
ndim = 3 # 2D problem (nothing to do with the dimension of the field!)
nTime = 1
nloc = 4 #  

value_mesh = np.zeros((nscalar,nNodes,nTime)) #value_mesh(nscalar,nonods,ntime)
value_mesh[0,:,0] = tracer[:,0] # streamwise velocity 

###################### other tests ####################################
# overwrite velocity field
#value_mesh[0,:,0] = 2
#value_mesh[0,:,0] = coordinates[:,1] #+ coordinates[:,0]
#######################################################################

# rectangular domain so
x0 = min(coordinates[:,0])
y0 = min(coordinates[:,1])
z0 = min(coordinates[:,2])
xN = max(coordinates[:,0])
yN = max(coordinates[:,1])
zN = max(coordinates[:,2])
print('(x0,y0,z0)',x0, y0, z0)
print('(xN,yN,zN)',xN, yN, zN)
block_x_start = np.array(( x0, y0, z0 )) 
#block_x_start = np.array(( 0, 0, 10 )) 

# get global node numbers
x_ndgln = np.zeros((nEl*nloc), dtype=int)
for iEl in range(nEl):
    n = vtu_data.GetCellPoints(iEl) + 1
    x_ndgln[iEl*nloc:(iEl+1)*nloc] = n


## set grid size
nx = 128#512#128
ny = 128#512#128
nz = 32#128#32
dx = (xN - x0) / nx
dy = (yN - y0) / ny
dz = (zN - z0) / nz
ddx = np.array([dx, dy, dz])

print('nx, ny, nz', nx, ny, nz)
print('dx, dy, dz', ddx)
#######################################################################
print('np.max(value_mesh)', np.max(value_mesh))
print('np.min(value_mesh)', np.min(value_mesh))

# interpolate from (unstructured) mesh to (structured) grid
# old code (value_grid, value_mesh, x_all,x_ndgln,ddx,block_x_start, nx,ny,nz, totele,nloc,nonods,nscalar,ndim,ntime) 
# new code (value_grid, value_mesh, x_all,x_ndgln,ddx,block_x_start, nx,ny,nz, ireturn_zeros_outside_grid, totele,nloc,nonods,nscalar,ndim,ntime)
zeros_outside_mesh = 1
value_grid = u2r.simple_interpolate_from_mesh_to_grid(value_mesh,x_all,x_ndgln,ddx,block_x_start,nx,ny,nz,zeros_outside_mesh,nEl,nloc,nNodes,nscalar,ndim,nTime) 

print('np.max(value_grid)', np.max(value_grid))
print('np.min(value_grid)', np.min(value_grid))

zeros_outside_mesh = 0
# old  (value_mesh, value_grid, block_x_start, ddx, x_all, nscalar, nx,ny,nz, nonods,ndim,ntime)
# new (value_mesh, value_grid, block_x_start, ddx, x_all, ireturn_zeros_outside_mesh, nscalar, nx,ny,nz, nonods,ndim,ntime) 
value_remesh = u2r.interpolate_from_grid_to_mesh(value_grid, block_x_start, ddx, x_all, zeros_outside_mesh,  nscalar,nx,ny,nz,nNodes,ndim,nTime)

print('np.max(value_remesh)', np.max(value_remesh))
print('np.min(value_remesh)', np.min(value_remesh))

print(value_remesh.shape)
#directory_Model = '/home/caq13/Inhale/data/'
filename = 'LSBU_' + str(i) + '.vtu'
ug=vtktools.vtu(filename)
ug.AddScalarField('Tracer_reconstructed', np.squeeze(value_remesh))
ug.Write('LSBU_' + str(i) + '.vtu')

