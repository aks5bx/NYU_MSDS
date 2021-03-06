#%%
###############################
### FUNCTIONS AND LIBRARIES ###
###############################

import os
import numpy as np
import random


def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings.
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on',
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines)
    words = filter(None, words)
    return list(words)


def load_and_shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    pos_path = "data/pos"
    neg_path = "data/neg"

    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)

    review = pos_review + neg_review
    random.shuffle(review)
    return review

# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

# %%
####################
### READ IN DATA ###
####################

reviews = load_and_shuffle_data()
reviews[0]

# %%
##################
### QUESTION 6 ###
##################

from collections import Counter

def sparse_bag_of_words(review_list):
    review_list = review_list[:-1]
    res_dict = Counter(review_list)

    return res_dict

# %%
##################
### QUESTION 7 ###
##################

X_train = []
y_train = []
for i in range(1500):
    data_slice = reviews[i]
    text = data_slice[:-1]
    label = data_slice[-1]

    feature_dict = sparse_bag_of_words(text)
    X_train.append(feature_dict)
    y_train.append(label)

X_test = []
y_test = []
for i in range(500):
    data_slice = reviews[i+1500]
    text = data_slice[:-1]
    label = data_slice[-1]

    feature_dict = sparse_bag_of_words(text)
    X_test.append(feature_dict)
    y_test.append(label)

# %%
#################
### PROBLEM 8 ###
#################

import time

start_time = time.time()

lambda_val = 0.01
t = 0
w = {}
termination_condition = False 
loop_counter = 0
order = np.arange(0, len(X_train), 1)

while not termination_condition:
    for i in order:
        t += 1
        n_t = 1 / (t * lambda_val)

        if y_train[i] * dotProduct(w, X_train[i]) < 1:
            
            part1 = {}
            increment(part1, (1 - (n_t * lambda_val)), w)
            increment(part1, n_t * y_train[i], X_train[i])

            w = part1

        else:
            part1 = {}
            increment(part1, (1 - (n_t * lambda_val)), w)
            w = part1

    random.shuffle(order)

    loop_counter += 1

    if loop_counter == 10:
        termination_condition = True

baseline_approach = time.time() - start_time

print("--- %s seconds ---" % (baseline_approach))

w

# %%
#################
### PROBLEM 9 ###
#################


def pegasos_sw(X_train, y_train, lambda_val, epochs, loud):
    start_time = time.time()

    lambda_val = lambda_val
    t = 1
    w = {}
    termination_condition = False 
    loop_counter = 0
    order = np.arange(0, len(X_train), 1)
    s = 1

    while not termination_condition:
        for i in order:
            t += 1
            n_t = 1 / (t * lambda_val)
            s = (1 - (n_t * lambda_val)) * s

            if y_train[i] * dotProduct(w, X_train[i]) < 1:
                '''
                part1 = {}
                increment(part1, (1 - (n_t * lambda_val)), w)
                increment(part1, n_t * y_train[i], X_train[i])
                w = part1
                '''

                increment(w, n_t * y_train[i] / s, X_train[i])

        random.shuffle(order)

        loop_counter += 1

        if loop_counter == epochs:
            termination_condition = True


    new_w = {}
    increment(new_w, s, w)

    sW_approach = time.time() - start_time

    if loud:
        print("--- %s seconds ---" % (sW_approach))

    return new_w

pegasos_sw(X_train, y_train, 0.01, 10, True)

# %%
##################
### PROBLEM 10 ###
##################

print('COMPARING RUN TIMES FOR 10 EPOCHS')
print('TIME FOR BASELINE RUN :', baseline_approach, ' SECONDS')
print('TIME FOR sW APPROACH RUN :', sW_approach, ' SECONDS')
# %%
##################
### PROBLEM 11 ###
##################

def classification_error(w, X, y):
    errors = 0
    num_iter = len(X)

    for i in range(num_iter):
        wx = dotProduct(w, X[i])

        if wx < 0:
            y_pred = -1
        else:
            y_pred = 1

        if y_pred != y[i]:
            errors += 1
    
    error_rate = errors / num_iter

    return error_rate

## Sample Run
classification_error(w, X_test, y_test)

# %%
##################
### PROBLEM 12 ###
##################

## First Pass
lambdas = [0.0001, 0.001, 0.01, 0.1]
errors = []

for lambda_val in lambdas:
    w = pegasos_sw(X_train, y_train, lambda_val, 10, False)
    error = classification_error(w, X_test, y_test)
    errors.append(error)

min_ind = errors.index(min(errors))
best_lambda = lambdas[min_ind]
best_lambda

#%%
## Best lambda is between 0.01 and 0.001 (discerned after multiple trials)
## Second Pass

lambdas = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
errors = []

for lambda_val in lambdas:
    w = pegasos_sw(X_train, y_train, lambda_val, 10, False)
    error = classification_error(w, X_test, y_test)
    errors.append(error)

min_ind = errors.index(min(errors))
best_lambda = lambdas[min_ind]

print('BEST LAMBDA VALUE :', best_lambda)
print('ERROR :', min(errors))

# %%