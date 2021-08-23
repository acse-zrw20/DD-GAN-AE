## Preprocessing

In this folder we use some legacy code for preprocessing. This only has to be ran once. Please use python 2 with numpy installed for executing the following commands.

### Usage

First thing to do is to descend into preprocessing source directory

```cd preprocessing/src```

and install the requirements

```pip install -r requirements.txt```

To compile the fortran library for python to use (u2r is the library):

```python2 -m numpy.f2py -c unstruc_mesh_2_regular_grid_new.f90 -m u2r```

To get POD coefficients execute the following to see the usage of the script that wraps Fortran:

```python2 get_pod_coeffs.py -h```


### Some common problems and fixes:

- An error along the lines of "ImportError: No module named vtkIOParallelPython"

Go to that line where that module is imported (the error stack will tell you) and comment it out. You are essentially editing your vtk package so only do this temporarily or if you don't care about the package.

- When using a virtualenv or conda environment

You need to replace line 4 with where the site-packages folder is for your environment
