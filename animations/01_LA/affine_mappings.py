import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the rotation matrix (90-degree counterclockwise)
theta = np.pi / 4  # 45 degrees
A = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# Define translation vector
translation = np.array([3, 2])

# Create a set of points forming a square
square = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])  # Close the square

# Apply linear transformation (rotation)
linear_transformed = np.dot(square, A.T)

# Apply affine transformation (rotation + translation)
affine_transformed = linear_transformed + translation

# Setup the plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-3, 5)
ax.set_ylim(-3, 5)
ax.set_aspect('equal')

# Plot the original square
original_line, = ax.plot([], [], 'bo-', label="Original")
linear_line, = ax.plot([], [], 'go-', label="Linear (Rotated)")
affine_line, = ax.plot([], [], 'ro-', label="Affine (Rotated + Translated)")

ax.legend()

# Animation function
def update(frame):
    t = frame / 20  # Interpolation factor (0 to 1)
    
    # Interpolate between original and transformed positions
    linear_step = (1 - t) * square + t * linear_transformed
    affine_step = (1 - t) * square + t * affine_transformed

    # Update the plot lines
    original_line.set_data(square[:, 0], square[:, 1])
    linear_line.set_data(linear_step[:, 0], linear_step[:, 1])
    affine_line.set_data(affine_step[:, 0], affine_step[:, 1])

    return original_line, linear_line, affine_line

# Create animation
ani = animation.FuncAnimation(fig, update, frames=20, interval=100, blit=True)

plt.show()
