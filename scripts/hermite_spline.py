"""Example for a hermite spline plotted in 3D."""
import numpy as np
from matplotlib import pyplot as plt

ax = plt.axes(projection="3d")

P_0 = np.array([1, 3, 4])
v_0 = np.array([0.5, 1, 0])
P_1 = np.array([5, 2, 4])
v_1 = np.array([0, -1, 0.5])

b = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [-3, -2, 3, -1], [2, 1, -2, 1]])

c = np.array([P_0, v_0, P_1, v_1])

points = np.array((P_0, P_1)).T
vel = np.array((v_0, v_1)).T
origin = np.zeros((3, 2))

# Draw begin and endpoint
ax.scatter(points[0], points[1], points[2], "bo")

# Draw velocity tangents
ax.quiver(*P_0.T, v_0[0], v_0[1], v_0[2])
ax.quiver(*P_1.T, v_1[0], v_1[1], v_1[2])

cache = b @ c

r = np.ones((101,) + cache.shape)[:] * cache
t = np.repeat((np.arange(101) / 100), 4).reshape(101, 4)

t[:, 0] = 1
t[:, 2] = np.power(t[:, 2], 2)
t[:, 3] = np.power(t[:, 3], 3)

# Slow but at least it works
res = np.array([t[i] @ r[i] for i in range(101)]).T

# Draw spline as dots
ax.scatter(res[0], res[1], res[2])

plt.show()
