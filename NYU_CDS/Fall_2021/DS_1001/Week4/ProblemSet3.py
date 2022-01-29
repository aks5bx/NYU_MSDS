#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 16:13:50 2021

@author: adisrikanth
"""

#%%
import pandas as pd

data = pd.read_csv('movieRatingsDeidentified.csv')[['As good as it gets (1997)', 'Magnolia (1999)']]
data.columns = ['good', 'mag']
#%%
##############
### T TEST ###
##############

import scipy
from scipy import stats

data = data.apply(pd.to_numeric, errors='coerce').fillna(data)

good = data[data.good.notnull()]['good']
good = good[good != " "]

mag = data[data.mag.notnull()]['mag']
mag = mag[mag != " "]


print(stats.ttest_ind(good, mag))

#%%
######################
### MANN WHITNEY U ###
######################

from scipy.stats import mannwhitneyu
results = mannwhitneyu(good, mag)
print(results)

#nx, ny = len(good), len(mag)
#U2 = nx*ny - U1
#rint(U2)


#%%
################
### ROW WISE ###
################

#%%
import pandas as pd

data = pd.read_csv('movieRatingsDeidentified.csv')[['As good as it gets (1997)', 'Magnolia (1999)']]
data = data.dropna()
data.columns = ['good', 'mag']
data = data.apply(pd.to_numeric, errors='coerce').fillna(data)
data = data[data.good != " "]
data = data[data.mag != " "]


#%%
##############
### T TEST ###
##############

data['good'] = data['good'].astype('float')
data['mag'] = data['mag'].astype('float')


print(stats.ttest_rel(data.good, data.mag))

#%%
######################
### MANN WHITNEY U ###
######################


from scipy.stats import mannwhitneyu
results = mannwhitneyu(data.good, data.mag)
print(results)

#nx, ny = len(data.good), len(data.mag)
#U2 = nx*ny - U1
#print(U2)



#%%
############
### LOTR ###
############

import pandas as pd

data = pd.read_csv('movieRatingsDeidentified.csv')[['The Lord of the Rings: The Fellowship of the Ring (2001)',
 'The Lord of the Rings: The Return of the King (2003)',
 'The Lord of the Rings: The Two Towers (2002)']]
data.columns = ['one', 'three', 'two']

data = data.dropna()
data = data.apply(pd.to_numeric, errors='coerce').fillna(data)
data = data[data.one != " "]
data = data[data.three != " "]
data = data[data.two != " "]

fvalue, pvalue = stats.f_oneway(data['one'], data['three'], data['two'])
print(fvalue, pvalue)



#%%
######################
### KRUSKAL WALLIS ###
######################

print(stats.kruskal(data['one'], data['three'], data['two']))



#%%
##################
### PROBLEM 20 ###
##################

data = pd.read_csv('movieRatingsDeidentified.csv')[['Gigli (2002)', 'The Shawshank Redemption (1994)']]
data.columns = ['gig', 'shaw']

data = data.apply(pd.to_numeric, errors='coerce').fillna(data)

gig = data[data.gig.notnull()]['gig']
gig = gig[gig != " "]

shaw = data[data.shaw.notnull()]['shaw']
shaw = shaw[shaw != " "]

results = mannwhitneyu(gig, shaw)
print(results)















