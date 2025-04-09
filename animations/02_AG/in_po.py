# ============================================
# 🔍 INNER PRODUCTS & GEOMETRIC INTERPRETATION
# ============================================

# -- BASIC IDEA:
# Inner products define "length" and "angle" in a vector space.
# The most common one? Dot product: ⟨x, y⟩ = xᵀy
# But we can generalize it with a matrix: ⟨x, y⟩_A = xᵀAy

# -- IN THE ANIMATION:
# Blue circle:
#   Set of vectors x such that xᵀx = 1
#   => All unit vectors under normal Euclidean geometry
#
# Red ellipse:
#   Set of vectors x such that xᵀAx = 1
#   => All unit vectors under a *new* inner product (defined by matrix A)
#   => A changes the geometry: it warps/stretch/rotates space

# -- GEOMETRIC TAKEAWAY:
# The red ellipse is the "unit circle" in the new space.
# Instead of ⟨x, x⟩ = ||x||² = 1, we now use ⟨x, x⟩_A = xᵀAx = 1.
# So what *used to be* a circle is now *an ellipse*.
#
# A shows how space is scaled and rotated.
# For example, a diagonal A matrix just scales axes.
# A non-diagonal A matrix will rotate/stretch the circle into an ellipse.

# -- VISUAL INTUITION:
# You're watching Euclidean space get *redefined* by matrix A.
# Blue: standard geometry.
# Red: transformed geometry under A.
# This is how inner products change what "length", "angle", and "orthogonality" mean.

# Pro tip: If A = Identity, the red ellipse = blue circle (aka default geometry).




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# define a general inner product using a symmetric positive-definite matrix A
A = np.array([[2, 0.5], [0.5, 1]])  # changes the "geometry" of space

# define inner product
def inner_product(x, y, A):
    return x.T @ A @ y

# define unit circle under standard and inner product norms
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])  # standard unit circle

# transform circle with respect to A-based inner product (like a distortion)
L = np.linalg.cholesky(A)
transformed_circle = L @ circle

# setup animation
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Euclidean (dot product)', color='blue')
line2, = ax.plot([], [], label='General Inner Product', color='red')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def animate(i):
    idx = i % len(theta)
    line1.set_data(circle[0, :idx], circle[1, :idx])
    line2.set_data(transformed_circle[0, :idx], transformed_circle[1, :idx])
    return line1, line2

ani = FuncAnimation(fig, animate, init_func=init, frames=len(theta), interval=20, blit=True)
plt.show()
