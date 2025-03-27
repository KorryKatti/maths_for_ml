import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the transformation matrix (R^3 to R^2)
A = np.array([[1, -1, 2], [2, 0, -1]])  # 2x3 matrix

# Generate random points in R^3 (image of V)
np.random.seed(42)
vectors = np.random.randn(100, 3)  # 100 points in R^3

# Transform them using A to get the image of Phi (subset of R^2)
transformed_vectors = vectors @ A.T  # Matrix multiplication

# Create figure
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0] = fig.add_subplot(121, projection='3d')  # Create 3D subplot
ax[0].set_title("Image of V (R^3)")
ax[0].set_xlim(-3, 3)
ax[0].set_ylim(-3, 3)
ax[0].set_zlim(-3, 3)
ax[1].set_title("Image of Phi (R^2)")
ax[1].set_xlim(-5, 5)
ax[1].set_ylim(-5, 5)

# 3D scatter plot for Image of V
ax[0] = fig.add_subplot(121, projection='3d')
scatter_v = ax[0].scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], color='b', alpha=0.5)
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].set_zlabel("Z")

# 2D scatter plot for Image of Phi
scatter_phi = ax[1].scatter([], [], color='r', alpha=0.5)
ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")

# Animation function
def update(frame):
    scatter_phi.set_offsets(transformed_vectors[:frame, :])
    return scatter_phi,

ani = animation.FuncAnimation(fig, update, frames=len(vectors), interval=50, repeat=True)
plt.show()
