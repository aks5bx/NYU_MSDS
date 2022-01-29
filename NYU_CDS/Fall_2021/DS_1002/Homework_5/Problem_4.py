
#%%
### IMPORT LIBRARIES, INITIALIZE DATA ###

import pandas as pd
import numpy as np 
import matplotlib.pyplot
import pylab

uniform_sample = np.random.uniform(0,1,1000)

# %%
### GENERATE X ###
def inverseCDF(u):
    if u <= 0.5:
        return (u / 2) ** 0.5
    else:
        u = u - 0.5
        return 1 - (u / 2) ** 0.5

### GENERATE Y ### 
def compute_y(x):
    if x <= 0.5:
        val = np.random.uniform(0,2*x,1)[0]

    else:
        val = np.random.uniform(0,(2 - (2*x)),1)[0]

    return val

#%%
### GENERATE X,Y PAIRS ###
x_values = []
y_values = []

for i in range(1000):
    u = uniform_sample[i]
    x = inverseCDF(u)
    y = compute_y(x)

    x_values.append(x)
    y_values.append(y)

matplotlib.pyplot.scatter(x_values,y_values)


# %%
