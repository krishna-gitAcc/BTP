import numpy as np
import math
f_vector = np.zeros([4, 1])
dof = [2, 3];
vec = [[2], [3]]
f_vector[dof, :]  += vec
print(f_vector)