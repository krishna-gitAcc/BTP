import numpy as np


def boundary_data_generator(nx, ny, n_side):
    """Function to generate the data for boundary conditions
    Inputs:
    nx = no. of elements in x direction
    ny = no. of elements in y direction
    n_side = no. of sides per element (4 - quadrilateral)"""

    n_ele = nx*ny
    bc_type = np.zeros((n_ele, n_side), dtype="int64")

    # Assign bottom boundary
    bc_type[:nx, 0] = 1
    # Assign right boundary
    bc_type[(nx-1):n_ele:nx, 1] = 2
    # Assign top boundary
    bc_type[nx*(ny-1):, 2] = 3
    # Assign left boundary
    bc_type[0::nx, 3] = -1

    return bc_type

def boundary_data_generator_plat_with_hole(nx, ny, n_side):

    """
    Inputs:
    nx = no. of elements in X-direction.
    ny = no. of elements in Y-direction.
    n_side = no. of sides per element (4-quadrilateral)
    """

    n_ele = 2*nx*ny
    bc_type = np.zeros((n_ele, n_side), dtype = "int64")

    #Assign bottom and top boundary  #
    bc_type[:nx, 0] = -1
    bc_type[2*nx*ny-nx:,2] = -1
    #Assign left boundary
    bc_type[::nx, 3] = 1
    #Assigning the right boundary
    #with boundary force
    bc_type[nx-1:nx*ny:nx, 1] = 2
    #without boundary force
    bc_type[nx*ny+nx-1::nx,1] = 3
    return bc_type



