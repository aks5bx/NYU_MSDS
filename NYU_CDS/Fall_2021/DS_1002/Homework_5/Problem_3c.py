
#%%
import numpy as np

## Plane One
p1 = np.array([4.75, 4.75, 0])
p2 = np.array([5.25, 5.25, 12/301])
p3 = np.array([10.25, 4.75, 0])

## Plane Two
p1 = np.array([4.75, 4.75, 0])
p2 = np.array([5.25, 5.25, 12/301])
p3 = np.array([4.75, 10.25, 0])

# These two vectors are in the plane
v1 = p3 - p1
v2 = p2 - p1

# the cross product is a vector normal to the plane
cp = np.cross(v1, v2)
a, b, c = cp

# This evaluates a * x3 + b * y3 + c * z3 which equals d
d = np.dot(cp, p3)

print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
# %%
