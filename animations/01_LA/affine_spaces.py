# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt

# Define vector subspace (a line through the origin)
t = np.linspace(-2, 2, 100)
subspace_x = t
subspace_y = t  # Line y = x (vector subspace through origin)

# Define affine subspace (same line but shifted)
x0, y0 = 2, 3  # Support point
affine_x = x0 + t
affine_y = y0 + t  # Shifted line y = x, but offset

# Plot
plt.figure(figsize=(6,6))
plt.axhline(0, color='black', linewidth=1)  # x-axis
plt.axvline(0, color='black', linewidth=1)  # y-axis

plt.plot(subspace_x, subspace_y, 'b--', label="Vector Subspace (Through Origin)")
plt.plot(affine_x, affine_y, 'r-', label="Affine Subspace (Shifted)")
plt.scatter([x0], [y0], color='red', marker='o', label="Support Point $(x_0)$")

# Labels and legend
plt.xlim(-3, 5)
plt.ylim(-3, 5)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.title("Vector Subspace vs Affine Subspace")
plt.grid()
plt.show()
