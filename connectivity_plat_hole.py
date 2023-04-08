import numpy as np
def connectivity(nx, ny):
    nel = 2*nx*ny
    connect = np.zeros((nel, 4), dtype = 'int64')
    el = 0;
    for i in range(2*ny):
        for j in range(nx):
            nd1 = j+(nx+1)*i
            nd2 = nd1+1
            nd3 = nd2+(nx+1)
            nd4 = nd3-1
            connect[el, :] = np.array([nd1, nd2, nd3, nd4])
            el = el+1
    return connect
