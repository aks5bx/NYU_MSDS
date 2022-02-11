#%%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#%%
def pre_process_mnist_01():
    """
    Load the mnist datasets, selects the classes 0 and 1 
    and normalize the data.
    Args: none
    Outputs: 
        X_train: np.array of size (n_training_samples, n_features)
        X_test: np.array of size (n_test_samples, n_features)
        y_train: np.array of size (n_training_samples)
        y_test: np.array of size (n_test_samples)
    """
    X_mnist, y_mnist = fetch_openml('mnist_784', version=1, 
                                    return_X_y=True, as_frame=False)
    indicator_01 = (y_mnist == '0') + (y_mnist == '1')
    X_mnist_01 = X_mnist[indicator_01]
    y_mnist_01 = y_mnist[indicator_01]
    X_train, X_test, y_train, y_test = train_test_split(X_mnist_01, y_mnist_01,
                                                        test_size=0.33,
                                                        shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.transform(X_test)

    y_test = 2 * np.array([int(y) for y in y_test]) - 1
    y_train = 2 * np.array([int(y) for y in y_train]) - 1
    return X_train, X_test, y_train, y_test

#%%
def sub_sample(N_train, X_train, y_train):
    """
    Subsample the training data to keep only N first elements
    Args: none
    Outputs: 
        X_train: np.array of size (n_training_samples, n_features)
        X_test: np.array of size (n_test_samples, n_features)
        y_train: np.array of size (n_training_samples)
        y_test: np.array of size (n_test_samples)
    """
    assert N_train <= X_train.shape[0]
    return X_train[:N_train, :], y_train[:N_train]

#%%
def classification_error(clf, X, y):
    preds = clf.predict(X)
    differences = np.sum(preds != y)    

    error = differences / len(preds)
    return error

#%%
X_train, X_test, y_train, y_test = pre_process_mnist_01()

clf = SGDClassifier(loss='log', max_iter=1000, 
                    tol=1e-3,
                    penalty='l1', alpha=0.01, 
                    learning_rate='invscaling', 
                    power_t=0.5,                
                    eta0=0.01,
                    verbose=1)
clf.fit(X_train, y_train)

#%%
##################
### PROBLEM 28 ###
##################

test = classification_error(clf, X_test, y_test)
train = classification_error(clf, X_train, y_train)
print('train: ', train, end='\t')
print('test: ', test)
# %%
##################
### PROBLEM 29 ###
##################

X_train, X_test, y_train, y_test = pre_process_mnist_01()
X_train, y_train = sub_sample(100, X_train, y_train)

alphas = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]
means = []
stds = []
for alpha_val in alphas:
    class_errors = []
    clf = SGDClassifier(loss='log', max_iter=1000, 
                tol=1e-3,
                penalty='l1', alpha=alpha_val, 
                learning_rate='invscaling', 
                power_t=0.5,                
                eta0=0.01,
                verbose=1)
    for i in range(0,10):
        clf.fit(X_train, y_train)
        test = classification_error(clf, X_test, y_test)
        class_errors.append(test)

    std_val = np.std(np.array(class_errors))
    mean_val = np.mean(class_errors)

    stds.append(std_val)
    means.append(mean_val)

# %%
plt.figure(0)
plt.errorbar(alphas, means, xerr = stds,fmt='o',ecolor = 'red')
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_xscale('log')
plt.title('Average Classification Error (Y Axis) Vs Alpha Value (X Axis) with Error (Stdev)')

# %%
##################
### PROBLEM 32 ###
##################

X_train, X_test, y_train, y_test = pre_process_mnist_01()
X_train, y_train = sub_sample(100, X_train, y_train)

for alpha_val in alphas: 
    clf = SGDClassifier(loss='log', max_iter=1000, 
                        tol=1e-3,
                        penalty='l1', alpha=alpha_val, 
                        learning_rate='invscaling', 
                        power_t=0.5,                
                        eta0=0.01,
                        verbose=1)
    clf.fit(X_train, y_train)

    coef = clf.coef_
    coef_reshape = coef.reshape((28, 28))

    scale = np.abs(clf.coef_).max()

    plt.figure()
    plt.imshow(coef_reshape, cmap=plt.cm.RdBu, vmax=scale, vmin=-scale)
    plt.colorbar()
    plt.title('Alpha = ' + str(alpha_val))

# %%
