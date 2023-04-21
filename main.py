import numpy as np

# given matrix K and arrays A and B
K = np.zeros((8, 8))  # initialize K with zeros
A = np.array([2, 4, 5, 6])
B = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [8, 7, 6, 5]])

# put values from B into K
K[A[:, np.newaxis], A] += B
K[A[:, np.newaxis], A] += B
print(K)