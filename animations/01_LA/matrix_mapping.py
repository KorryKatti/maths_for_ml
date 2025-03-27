import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Define a random transformation matrix from R^3 to R^2
A = np.array([[1, -2, 0.5], [0, 1, -1]])  # 2x3 matrix

# Generate random points in R^3
num_points = 100
X = np.random.uniform(-2, 2, (3, num_points))  # 3xN points

# Apply transformation: Y = A * X
Y = A @ X  # Resulting 2xN points in R^2

fig = plt.figure(figsize=(10, 5))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)
ax2d.set_xlim(-5, 5)
ax2d.set_ylim(-5, 5)

# Plot initial points in 3D
scatter3d = ax3d.scatter(X[0], X[1], X[2], c='b', label="Original Points")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.set_title("Original Points in R^3")
ax3d.legend()

# Plot transformed points in 2D
scatter2d, = ax2d.plot([], [], 'ro', label="Transformed Points")
ax2d.set_xlabel("X")
ax2d.set_ylabel("Y")
ax2d.set_title("Mapped Points in R^2")
ax2d.legend()

# Animation function
def update(frame):
    if frame < num_points:
        scatter2d.set_data(Y[0, :frame], Y[1, :frame])
    return scatter2d,

ani = animation.FuncAnimation(fig, update, frames=num_points, interval=50, blit=True)
plt.show()
