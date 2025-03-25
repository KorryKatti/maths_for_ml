"""
this animation shows a plane in 3D (a vector subspace) and vectors within it. each frame adds a new vector to show:

two initial vectors in the plane

their sum (which stays in the plane)

a scaled version of one vector (also stays in the plane)

this should help you see why a plane in 3D can be a vector subspace—it’s closed under addition and scalar multiplication.

"""


"""
imagine you’re in a big city, and the entire city is the **vector space** (3D). now, think of a specific subway system that only moves on tracks—this is a **vector subspace** (2D).  

### why?  
- if you take two valid subway routes and combine them (vector addition), you're still in the subway system.  
- if you take one route and travel it at half speed or double speed (scalar multiplication), you're still on the subway.  
- but if you decide to fly a helicopter instead, you leave the subway system → not in the subspace anymore.  

the idea: **a vector subspace is a smaller space that stays closed under all vector operations (addition & scaling).**

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Define the plane (a vector subspace in 3D)
def plane(x, y):
    return 0.5 * x + 0.2 * y  # Example plane equation

# Create figure and 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Generate a grid of points for the plane
x_vals = np.linspace(-2, 2, 10)
y_vals = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x_vals, y_vals)
Z = plane(X, Y)

# Initial plot of the plane
ax.plot_surface(X, Y, Z, color='cyan', alpha=0.5)

# Define initial vectors in the plane
v1 = np.array([1, 1, plane(1, 1)])
v2 = np.array([-1, 2, plane(-1, 2)])

# Compute vector addition and scalar multiplication
v_sum = v1 + v2
v_scaled = 1.5 * v1

# Prepare animation data
vectors = [v1, v2, v_sum, v_scaled]
colors = ['r', 'g', 'b', 'm']
labels = ['v1', 'v2', 'v1 + v2', '1.5 * v1']

# Function to update animation frames
def update(frame):
    ax.clear()
    ax.plot_surface(X, Y, Z, color='cyan', alpha=0.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("Vector Subspace Animation")

    # Plot vectors dynamically
    for i in range(frame + 1):
        vec = vectors[i]
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=colors[i], label=labels[i])

    ax.legend()

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(vectors), interval=1000, repeat=True)
plt.show()
