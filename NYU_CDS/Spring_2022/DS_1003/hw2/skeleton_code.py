#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%%
#######################################
### Feature normalization
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size(num_instances, num_features)
        test - test set, a 2D numpy array of size(num_instances, num_features)

    Returns:
        train_normalized - training set after normalization
        test_normalized - test set after normalization
    """

    # Remove constant columns - create mask
    b = train == train[0,:]
    mask = b.all(axis=0)
    # Use mask to subset data 
    ret_train = train[:, ~mask]

    # Extract minimum row
    minimum_row_mask = np.argmin(ret_train, axis=0)
    minimum_row = ret_train.min(axis=0)
    
    # Extract maximum row
    maximum_row_mask = np.argmax(ret_train, axis=0)
    maximum_row = ret_train.max(axis=0)

    # Scale train data 
    # Subtract by minimum 

    ret_train = ret_train - minimum_row

    # Divide by range 
    ret_train = ret_train / (maximum_row - minimum_row)

    # Same process for test data 
    b = test == test[0,:]
    mask = b.all(axis=0)
    # Use mask to subset data 
    ret_test = test[:, ~mask]

    # Scale test data using train data statistics
    # Subtract by minimum 
    ret_test = ret_test - minimum_row

    # Divide by range 
    ret_test = ret_test / (maximum_row - minimum_row)

    return ret_train, ret_test

#%%
#######################################
### The square loss function
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D array of size(num_features)

    Returns:
        loss - the average square loss, scalar
    """
    #TODO

    m = X.shape[0]
    
    pt1 = theta @ X.transpose() @ X @ theta 
    pt2 = 2 * ((X @ theta).transpose() @ y)
    pt3 = y.transpose() @ y

    loss = (pt1 - pt2 + pt3) * (1/m)
    
    return loss

#%%
#######################################
### The gradient of the square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute the gradient of the average square loss(as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size(num_features)
    """
    #TODO

    m = X.shape[0]

    pt1 = X.T @ X @ theta 
    pt2 = X.T @ y

    res = (pt1 - pt2) * (2/m)

    return res

#%%
#######################################
### Gradient checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm. Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
(e_1 =(1,0,0,...,0), e_2 =(0,1,0,...,0), ..., e_d =(0,...,0,1))

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
(J(theta + epsilon * e_i) - J(theta - epsilon * e_i)) /(2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicating whether the gradient is correct or not
    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO


    # for each coordinate direction 
    for i in range(num_features):
        # Define our e_i, overwrite the 0 in the ith position with a 1
        epsilon_i = np.zeros(num_features)
        epsilon_i[i] = 1 

        # Compute approximation of the derivative
        approx1 = compute_square_loss(X, y, theta + (epsilon * epsilon_i))
        approx2 = compute_square_loss(X, y, theta - (epsilon * epsilon_i))
        approx_res = (approx1 - approx2) / (2 * epsilon)

        approx_grad[i] = approx_res

    # Compute euclidean distance between true and approxmiate
    euc_diff = np.linalg.norm(approx_grad-true_gradient)

    if euc_diff > tolerance: 
        return False
    else:
        return True

#%%
#######################################
### Generic gradient checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, 
                             epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. 
    And check whether gradient_func(X, y, theta) returned the true 
    gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO


#%%
#######################################
### Batch gradient descent
def batch_grad_descent(X, y, alpha=0.1, num_step=1000, grad_check=False):
    """
    In this question you will implement batch gradient descent to
    minimize the average square loss objective.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size(num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step(num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array,(num_step+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step + 1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_step + 1)  #Initialize loss_hist
    theta = np.zeros(num_features)  #Initialize theta
    #TODO

    for step in range(num_step+1):
        # Conduct step into gradient descent 
        sq_loss_gradient = compute_square_loss_gradient(X, y, theta)

        # Update tracked values
        theta_hist[step,:] = theta
        loss_hist[step] = compute_square_loss(X, y, theta)

        # Update theta
        theta -= (alpha * sq_loss_gradient)

    return theta_hist, loss_hist

#%%
#######################################
### The gradient of regularized batch gradient descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized average square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size(num_features)
    """
    #TODO

    m = X.shape[0]

    pt1 = X.T @ X @ theta 
    pt2 = X.T @ y

    res1 = (pt1 - pt2) * (2/m)
    res = res1 + (2 * lambda_reg * theta)

    return res

#%%
#######################################
### Regularized batch gradient descent
def regularized_grad_descent(X, y, alpha=0.05, lambda_reg=10**-2, num_step=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        num_step - number of steps to run
    
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size(num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step(num_step+1) is theta_hist[-1]
        loss hist - the history of average square loss function without the regularization term, 1D numpy array.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    #TODO

    for step in range(num_step+1):
        # Conduct step into gradient descent 
        sq_loss_gradient = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)

        # Update tracked values
        theta_hist[step,:] = theta
        loss_hist[step] = compute_square_loss(X, y, theta)

        # Update theta
        theta -= (alpha * sq_loss_gradient)

    return theta_hist, loss_hist

#######################################
### Stochastic gradient descent
def stochastic_grad_descent(X, y, alpha=0.01, lambda_reg=10**-2, num_epoch=1000, eta0=False):
    """
    In this question you will implement stochastic gradient descent with regularization term

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - string or float, step size in gradient descent
                NOTE: In SGD, it's not a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every step is the float.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t).
                if alpha == "1/t", alpha = 1/t.
        lambda_reg - the regularization coefficient
        num_epoch - number of epochs to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size(num_epoch, num_instances, num_features)
                     for instance, theta in epoch 0 should be theta_hist[0], theta in epoch(num_epoch) is theta_hist[-1]
        loss hist - the history of loss function vector, 2D numpy array of size(num_epoch, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta

    theta_hist = np.zeros((num_epoch, num_instances, num_features)) #Initialize theta_hist
    loss_hist = np.zeros((num_epoch, num_instances)) #Initialize loss_hist
    #TODO

#%%
def load_data():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('ridge_regression_dataset.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    return X_train, y_train, X_test, y_test

# %%
###########################
### PROBLEM TWELVE PLOT ###
###########################

X_train, y_train, X_test, y_test = load_data()

x_axis = np.arange(1, 1002, 1, dtype=int)
losses = []
steps = [0.5, 0.1, .05, 0.01]
for step in steps:
    x, y = batch_grad_descent(X_train, y_train, alpha=step, num_step=1000, grad_check=False)
    losses.append(y)

plt.figure(0)
plt.plot(x_axis, np.log(losses[0]), label = 'Alpha = 0.5')
plt.plot(x_axis, np.log(losses[1]), label = 'Alpha = 0.1')
plt.legend(loc="upper left")
plt.title('TRAIN Diverging Losses: Log Loss (Y Axis) vs Num Steps (X Axis)')


plt.figure(1)
plt.plot(x_axis, losses[2], label = 'Alpha = 0.05')
plt.plot(x_axis, losses[3], label = 'Alpha = 0.01')
plt.legend(loc="upper left")
plt.title('TRAIN Converging Losses: Loss (Y Axis) vs Num Steps (X Axis)')


# %%
#############################
### PROBLEM THIRTEEN PLOT ###
#############################

thetas, loss_vals = batch_grad_descent(X_train, y_train, alpha=0.05, num_step=1000)

avg_losses = []
for i in range (0,1001):
    avg_losses.append(compute_square_loss(X_test, y_test, thetas[i]))


plt.figure(0)
plt.plot(x_axis, avg_losses)
plt.title('TEST Loss (Y Axis) vs Iterations (X Axis)')

# %%
#########################
### PROBLEM SEVENTEEN ###
#########################

X_train, y_train, X_test, y_test = load_data()

x_axis = np.arange(1, 1002, 1, dtype=int)
losses = []
avg_losses_store = []
lambdas = [10**-7, 10**-5, 10**-3, 10**-1, 1, 10, 100] 
for lambda_val in lambdas[0:4]:
    x, y = regularized_grad_descent(X_train, y_train, alpha=0.05, lambda_reg=lambda_val, num_step=1000)
    losses.append(y)

    avg_losses = []
    for i in range (0,1001):
        avg_losses.append(compute_square_loss(X_test, y_test, x[i]))

    avg_losses_store.append(avg_losses)

    plt.figure(3)
    plt.plot(x_axis, avg_losses, label = str(lambda_val))
    plt.title('TEST Loss (Y Axis) vs Iterations (X Axis)')
    plt.legend(loc="upper left")

for lambda_val in lambdas[4:]:
    x, y = regularized_grad_descent(X_train, y_train, alpha=0.05, lambda_reg=lambda_val, num_step=1000)
    losses.append(y)

    avg_losses = []
    for i in range (0,1001):
        avg_losses.append(compute_square_loss(X_test, y_test, x[i]))

    avg_losses_store.append(avg_losses)

    plt.figure(4)
    plt.plot(x_axis, avg_losses, label = str(lambda_val))
    plt.title('TEST Loss (Y Axis) vs Iterations (X Axis)')
    plt.legend(loc="upper left")
    

plt.figure(5)
plt.plot(x_axis, np.log(losses[0]), label = 'Lambda = 10e-7')
plt.plot(x_axis, np.log(losses[1]), label = 'Lambda = 10e-5')
plt.plot(x_axis, np.log(losses[2]), label = 'Lambda = 10e-3')
plt.plot(x_axis, np.log(losses[3]), label = 'Lambda = 10e-1')
plt.legend(loc="upper left")
plt.title('TRAIN Converging Losses: Log Loss (Y Axis) vs Num Steps (X Axis)')

plt.figure(6)
plt.plot(x_axis, np.log(losses[4]), label = 'Lambda = 1')
plt.plot(x_axis, np.log(losses[5]), label = 'Lambda = 10')
plt.plot(x_axis, np.log(losses[6]), label = 'Lambda = 100')
plt.legend(loc="upper left")
plt.title('TRAIN Diverging Losses: Log Loss (Y Axis) vs Num Steps (X Axis)')

# %%
########################
### PROBLEM EIGHTEEN ###
########################

train_sq_losses = [losses[0][-1], losses[1][-1], losses[2][-1], losses[3][-1], losses[4][-1], losses[5][-1], losses[6][-1]]
test_sq_losses = [avg_losses_store[0][-1], avg_losses_store[1][-1], avg_losses_store[2][-1], avg_losses_store[3][-1], avg_losses_store[4][-1], avg_losses_store[5][-1], avg_losses_store[6][-1]]

plt.figure(7)
plt.scatter(lambdas, np.log(train_sq_losses), label = 'Train Losses')
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_xscale('log')
plt.title('Log Train Losses (Y Axis) Vs Lambda Value (X Axis)')


plt.figure(8)
plt.scatter(lambdas, np.log(test_sq_losses), label = 'Test Losses')
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_xscale('log')
plt.title('Log Test Losses (Y Axis) Vs Lambda Value (X Axis)')

'''
Based on the information I have from this and previous problems, 
I would choose the lambda = 0.01 
'''

# %%
########################
### PROBLEM NINETEEN ###
########################

test_sq_losses_min = [min(losses[0]), min(losses[1]), min(losses[2]), min(losses[3]), min(losses[4]), min(losses[5]), min(losses[6])]

plt.figure(9)
plt.scatter(lambdas, np.log(test_sq_losses), label = 'Test Losses (Last)')
plt.scatter(lambdas, np.log(test_sq_losses_min), label = 'Test Losses (Min)')
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_xscale('log')
plt.title('Log Test Losses, Log Test Losses (Min) (Y Axis) Vs Lambda Value (X Axis)')

plt.figure(10)
plt.scatter(lambdas, np.log(test_sq_losses_min), label = 'Test Losses (Min)')
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_xscale('log')
plt.title('Log Test Losses (Min) (Y Axis) Vs Lambda Value (X Axis)')

'''
The value I would select is the same as before
'''
# %%
