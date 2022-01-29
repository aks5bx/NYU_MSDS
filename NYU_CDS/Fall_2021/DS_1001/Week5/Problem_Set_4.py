
#%%
import pandas as pd
import random 
data = pd.read_csv('movieRatingsDeidentified.csv')

#%%

### PROBLEM 12 ###

data = pd.read_csv('movieRatingsDeidentified.csv')[['Good Will Hunting (1997)', 'Reservoir Dogs (1992)']]
data = data.dropna()
data.columns = ['good', 'dog']
data = data.apply(pd.to_numeric, errors='coerce').fillna(data)
data = data[data.good != " "]
data = data[data.dog != " "]

#%%
actualDiff = data.dog.mean() - data.good.mean()
# %%
## Sampling ##

### PROBLEM 15 ###
differences = []
for i in range(10000):
    totalList = list(data.dog) + list(data.good)
    random.shuffle(totalList)

    group1 = totalList[:len(totalList)//2]
    group2 = totalList[len(totalList)//2:]

    diff = (sum(group1) / len(group1)) - (sum(group2) / len(group2))
    differences.append(diff)


# %%
### PROBLEM 16 ###
data = pd.read_csv('movieRatingsDeidentified.csv')[['Unforgiven (1992)', 'Big Fish (2003)']]
data = data.dropna()
data.columns = ['good', 'dog']
data = data.apply(pd.to_numeric, errors='coerce').fillna(data)
data = data[data.good != " "]
data = data[data.dog != " "]

actualDiff = data.dog.mean() - data.good.mean()


differences = []
for i in range(10000):
    totalList = list(data.dog) + list(data.good)
    random.shuffle(totalList)

    group1 = totalList[:len(totalList)//2]
    group2 = totalList[len(totalList)//2:]

    diff = (sum(group1) / len(group1)) - (sum(group2) / len(group2))
    differences.append(diff)

greater_elements = len([i for i in differences if i > actualDiff])
greater_elements/10000
# %%
### PROBLEM 17 ###
data = pd.read_csv('movieRatingsDeidentified.csv')[['Pulp Fiction (1994)', 'Magnolia (1999)']]
data = data.dropna()
data.columns = ['good', 'dog']
data = data.apply(pd.to_numeric, errors='coerce').fillna(data)
data = data[data.good != " "]
data = data[data.dog != " "]

actualDiff = data.dog.mean() - data.good.mean()

differences = []
for i in range(10000):
    totalList = list(data.dog) + list(data.good)
    random.shuffle(totalList)

    group1 = totalList[:len(totalList)//2]
    group2 = totalList[len(totalList)//2:]

    diff = (sum(group1) / len(group1)) - (sum(group2) / len(group2))
    differences.append(diff)

greater_elements = len([i for i in differences if i > actualDiff])
greater_elements/10000
# %%
### PROBLEM 18 ###

data = pd.read_csv('movieRatingsDeidentified.csv')[['Mulholland Dr. (2001)']]
data = data.dropna()
data.columns = ['movie']
data = data.apply(pd.to_numeric, errors='coerce').fillna(data)
data = data[data.movie != " "]

means = []
for i in range (0, 10000): 
    sample = data.sample(n= len(data), replace=True, random_state=1).values.mean()
    means.append(sample)

count_val = len([i for i in means if i > 2])
count_val / 10000

# %%
### PROBLEM 19 ### 

data = pd.read_csv('movieRatingsDeidentified.csv')[['Zoolander (2001)']]
data = data.dropna()
data.columns = ['movie']
data = data.apply(pd.to_numeric, errors='coerce').fillna(data)
data = data[data.movie != " "]

means = []
for i in range (0, 10000): 
    sample = data.sample(n= len(data), replace=True).values.mean()
    means.append(sample)

count_val = len([i for i in means if i <= 2.5])
count_val / 10000
# %%
### PROBLEM 20 ### 

data = pd.read_csv('movieRatingsDeidentified.csv')[['Saving Private Ryan (1998)']]
data = data.dropna()
data.columns = ['movie']
data = data.apply(pd.to_numeric, errors='coerce').fillna(data)
data = data[data.movie != " "]

means = []
for i in range (0, 10000): 
    sample = data.sample(n= len(data), replace=True).values.mean()
    means.append(sample)

count_val = len([i for i in means if i > 3])
count_val / 10000
# %%
