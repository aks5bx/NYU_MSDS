#%%
import numpy as np
import matplotlib.pyplot as plt

def get_a(deg_true):
    """
    Inputs:
    deg_true: (int) degree of the polynomial g
    
    Returns:
    a: (np array of size (deg_true + 1)) coefficients of polynomial g
    """
    return 5 * np.random.randn(deg_true + 1)

def get_design_mat(x, deg):
    """
    Inputs:
    x: (np.array of size N)
    deg: (int) max degree used to generate the design matrix
    
    Returns:
    X: (np.array of size N x (deg_true + 1)) design matrix
    """
    X = np.array([x ** i for i in range(deg + 1)]).T
    return X

def draw_sample(deg_true, a, N):
    """
    Inputs:
    deg_true: (int) degree of the polynomial g
    a: (np.array of size deg_true) parameter of g
    N: (int) size of sample to draw
    
    Returns:
    x: (np.array of size N)
    y: (np.array of size N)
    """    
    x = np.sort(np.random.rand(N))
    X = get_design_mat(x, deg_true)
    y = X @ a
    return x, y

def draw_sample_with_noise(deg_true, a, N):  
    """
    Inputs:
    deg_true: (int) degree of the polynomial g
    a: (np.array of size deg_true) parameter of g
    N: (int) size of sample to draw
    
    Returns:
    x: (np.array of size N)
    y: (np.array of size N)
    """  
    x = np.sort(np.random.rand(N))
    X = get_design_mat(x, deg_true)
    y = X @ a + np.random.randn(N)
    return x, y

#%%
######################
### PRE PROCESSING ###
######################
a = get_a(2)

x_train, y_train = draw_sample(2, a, 10)
x_test, y_test = draw_sample(2, a, 100)

#%%
#####################
### PROBLEM SEVEN ###
#####################

def least_square_estimator(X, y):
    # Get rows (N) and columns (d)
    N = X.shape[0]
    d = X.shape[1]

    if d > N:
        print('N must be greater or equal to d')
        return None 

    # Otherwise, compute b = (XTX)-1 (XTy)
    XTX_inv = np.linalg.inv(X.T @ X)
    XTy = (X.T) @ y

    b_estimate = XTX_inv @ XTy
    return b_estimate

#%%
#####################
### PROBLEM EIGHT ###
#####################

def empirical_risk(X, y, b):

    # Get estimated values for y 

    y_est = X @ b
    # Compute differences per the L2 Norm
    sq_differences = (y_est - y) ** 2
    sum_sq_differences = np.sum(sq_differences)

    # Compute emp risk 
    emp_risk = sum_sq_differences / len(y_est)

    return emp_risk 


#%%
####################
### PROBLEM NINE ###
####################

x_train_9 = get_design_mat(x_train, 2)
x_test_9 = get_design_mat(x_test, 2)

b_est = least_square_estimator(x_train_9, y_train)

print('COMPARISON OF estimated b value and a')
print('estimated b vector: ', b_est)
print('true a vector: ', a)

#%%
x = np.linspace(0, 1, num=100)
g_x = []
fb_x = []
for i in range(100):
    g_val = a @ np.array([x[i] ** 0, x[i] ** 1, x[i] ** 2])
    g_x.append(g_val)

    fb_val = b_est @ np.array([x[i] ** 0, x[i] ** 1, x[i] ** 2])
    fb_x.append(g_val)


plt.scatter(x_train, y_train) 
plt.plot(x, fb_x, 'b', label = 'f_b(x)')
plt.plot(x, g_x, 'g')
plt.legend(labels=['g(x)', 'f_b(x)', 'training data'])
plt.show()
# %%
###################
### PROBLEM TEN ###
###################

for i in range(1,5):
    d = i
    x_train_10 = get_design_mat(x_train, d)
    x_test_10 = get_design_mat(x_test, d)

    b_est = least_square_estimator(x_train_10, y_train)

    print('d = ', i, ':', empirical_risk(x_test_10, y_test, b_est))

print('The minimum value at which we can get a near perfect fit is d = 2')

# %%
######################
### PROBLEM ELEVEN ###
######################



# %%

# %%
