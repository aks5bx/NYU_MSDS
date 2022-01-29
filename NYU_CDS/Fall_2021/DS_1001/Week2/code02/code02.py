#!/usr/bin/env python
# coding: utf-8

# ### Lab02 
# 
# ### 2021-09-09

# ### Moments: Examples: Functions for Mean and Variance in Python 

# In[1]:


import numpy as np


# In[2]:


def fn_mean_var(data):
    
    if len(data) == 0: 
        return (None, None)

    mean = sum(data) / len(data)
    var = sum((data - mean)** 2) / len(data)
    
    return (mean, var)



data = np.array((range(10)))
print(data)
print(fn_mean_var(data))


# In[3]:


def fn_mean_var_second(data):

    if len(data) == 0:
        return (None, None)

    mean = sum(data) / len(data)
    var = sum((data - mean) ** 2) / len(data)

    # return a dictionary object
    # so we have control of the output
    return {
        "mean": mean,
        "var": var,
    }


data = np.array((range(10)))
print(data)
print(fn_mean_var_second(data))


# In[4]:


# Always use a "standard" implementation if exists

def fn_mean_var_third(data):
    # the function translates data into a numpy array
    # and applies the methods for mean and var
    return {"mean": np.array(data).mean(), "var": np.array(data).var()}


data = list(range(10))
print(data)
print(fn_mean_var_third(data))


# ### Functions to calculate first 10 moments:
# 
# 
# Recall from lecture: 
# 
# $$M_k := E(X - E(X))^k$$

# In[5]:


def fn_moments(data, number_of_moments=10):

    if len(data) == 0:
        return None

    data = np.array(data)
    mean = data.mean()

    # return a dictionary: where k points to k-th moment
    return {k: np.array((data - mean) ** k).mean() for k in range(number_of_moments)}


data = list(range(10))
print("date: ", data)
print("momdents:", fn_moments(data))


# Examine the 0th moment; 
# 
# Examine the odd moments: What values do they take? Does it makes sense? 

# ### Generate random variables 

# In[6]:


### generate 1000 i.i.d. random variables X_i's:


### X_i follows unif[0, 1]
number_of_samples = 1000
sample_uniform = np.random.uniform(0, 1, number_of_samples)

# display first 10
sample_uniform[ :10]


# In[7]:


# examine our function
fn_mean_var_third(sample_uniform)


# In[8]:


### generate 1000 i.i.d. random variables X_i's:

### X_i follows normal N(0, 1)
number_of_samples = 1000
mu, sigma = 0, 1
sample_normal = np.random.normal(mu, sigma, number_of_samples)

# display first 10
sample_normal[ :10]


# In[9]:


# examine our function
fn_mean_var_third(sample_normal)


# ### plot histograms: 

# In[10]:


import matplotlib.pyplot as plt


# In[11]:



count, bins, ignored = plt.hist(sample_uniform, 15, density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()


# In[12]:


count, bins, ignored = plt.hist(sample_normal, 15, density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()


# ### CDF for Uniform Distribution

# In[13]:


### sample_uniform

sorted_random_data = np.sort(sample_uniform)
x = np.arange(len(sorted_random_data)) / float(len(sorted_random_data) - 1)

fig = plt.figure()
fig.suptitle('Uniform CDF: F(x) = P(X <= x)')
ax = fig.add_subplot(111)
ax.plot(sorted_random_data, x)
ax.set_xlabel('x')
ax.set_ylabel('')


# ### CDF for Normal Distribution

# In[14]:


### sample_normal

sorted_random_data = np.sort(sample_normal)
x = np.arange(len(sorted_random_data)) / float(len(sorted_random_data) - 1)

fig = plt.figure()
fig.suptitle('Normal CDF: F(x) = P(X <= x)')
ax = fig.add_subplot(111)
ax.plot(sorted_random_data, x)
ax.set_xlabel('x')
ax.set_ylabel('')


# In[ ]:




