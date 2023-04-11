
# from matplotlib import pyplot as plt
# import numpy as np

# # Define the plate geometry
# Lx = 1.0
# Ly = 1.0
# hole_radius = 0.2

# # Define the number of elements in each direction
# nelx = 1000
# nely = 1000

# # Create the nodal coordinates
# dx = Lx / nelx
# dy = Ly / nely
# coords = []
# for i in range((nelx//2)+1):
#     for j in range((nely//2)+1):
#         node = j*(nelx//2+1) + i
#         x = i*dx
#         y = j*dy
#         r = np.sqrt((x - hole_radius)**2 + (y - hole_radius)**2)
#         if r <= hole_radius:
#             coords.append([hole_radius + r*np.cos(np.pi/4), hole_radius + r*np.sin(np.pi/4)])
#         else:
#             coords.append([x, y])

# # Create the element connectivity matrix
# connectivity = []
# for j in range(nely//2):
#     for i in range(nelx//2):
#         if i == 0 and j == 0:
#             continue
#         element = j*(nelx//2) + i
#         n1 = j*(nelx//2+1) + i
#         n2 = n1 + 1
#         n3 = (j+1)*(nelx//2+1) + i + 1
#         n4 = n3 - 1
#         connectivity.append([n1, n2, n3, n4])

# # Convert lists to numpy arrays
# coords = np.array(coords)
# connectivity = np.array(connectivity, dtype=int)

# # 4 node quadrilateral
# xs = [x[0] for x in coords]
# ys = [x[1] for x in coords]
# plt.scatter(xs, ys)
# plt.show()

# print(connectivity)

import mesh_gen
import numpy as np
from matplotlib import pyplot as plt
mesh_obj = mesh_gen.platWithHole(64, 64, [2, 0, 7, 0, 7, 7, 0, 7, 0, 2], 0)
coord = mesh_obj.coord_array()
connectivity = mesh_obj.connectivity()