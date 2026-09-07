import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0.5, 1.5],
    [1, 1],
    [1.5, 0.5],
    [3, 0.5],
    [2, 2],
    [1, 2.5]
])

y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

# Decision boundary
x0 = np.arange(0, 6)
x1 = 3 - x0

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

# Decision boundary
ax.plot(x0, x1, c="b")

ax.axis([0, 4, 0, 3.5])

# Shade region
ax.fill_between(x0, x1, alpha=0.2)

# Plot y = 0 points
ax.scatter(X[:3, 0], X[:3, 1], marker='x', label='y = 0')

# Plot y = 1 points
ax.scatter(X[3:, 0], X[3:, 1], marker='o', label='y = 1')

ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
ax.legend()

plt.show()