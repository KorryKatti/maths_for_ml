import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define matrix A (mapping from R^3 to R^2)
A = np.array([[1, 2, -1],
              [0, 1, 3]])

# Compute the null space (kernel of A)
kernel = null_space(A)  # This finds the set of all x such that Ax = 0

print("Kernel (Null Space) basis vectors:")
print(kernel)

# Compute the column space (image of A)
column_space = A @ np.eye(A.shape[1])  # Just taking the columns of A

print("Column Space (Image) basis vectors:")
print(column_space)

# Plot the column space and kernel
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot kernel (null space)
kernel_vector = kernel[:, 0]
k_scale = np.linspace(-5, 5, 10)
k_points = np.outer(k_scale, kernel_vector).T
ax.plot(k_points[0], k_points[1], k_points[2], 'r', label='Kernel (Null Space)')

# Plot column space
col_vectors = column_space.T  # Each column is a basis vector
ax.quiver(0, 0, 0, col_vectors[0, 0], col_vectors[1, 0], 0, color='b', label='Column Space (Image)')
ax.quiver(0, 0, 0, col_vectors[0, 1], col_vectors[1, 1], 0, color='b')

# Labels and legend
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Kernel and Column Space Visualization')
ax.legend()

plt.show()
