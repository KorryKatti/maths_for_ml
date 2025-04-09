import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("L1 vs L2 Norm Unit Circles")

# Plot placeholders
l1_plot, = ax.plot([], [], 'r-', label='L1 Norm (Manhattan)', lw=2)
l2_plot, = ax.plot([], [], 'b-', label='L2 Norm (Euclidean)', lw=2)
ax.legend()

# Generate points for L1 and L2 unit circles
theta = np.linspace(0, 2 * np.pi, 400)
l2_x = np.cos(theta)
l2_y = np.sin(theta)

def l1_circle_points(resolution=400):
    # Parametric points around L1 unit circle
    points = []
    for t in np.linspace(0, 2*np.pi, resolution):
        x = np.cos(t)
        y = np.sin(t)
        norm = abs(x) + abs(y)
        if norm != 0:
            x /= norm
            y /= norm
        points.append((x, y))
    return np.array(points)

l1_points = l1_circle_points()

# Animation update function
def update(frame):
    l1_plot.set_data(l1_points[:frame, 0], l1_points[:frame, 1])
    l2_plot.set_data(l2_x[:frame], l2_y[:frame])
    return l1_plot, l2_plot

# Create animation
ani = animation.FuncAnimation(fig, update, frames=400, interval=10, blit=True)

# # Save the animation to a file
# ani.save('l1_vs_l2_norm_circles.mp4', writer='ffmpeg', fps=30)

plt.show()
