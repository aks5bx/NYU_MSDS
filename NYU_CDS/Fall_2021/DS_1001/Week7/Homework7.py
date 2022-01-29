#%%
import pandas as pd 

data = pd.read_csv('kepler.csv', header = None)
data.columns = ['caste', 'iq', 'brainmass', 'hours', 'income']

data
# %%
### PROBLEM ONE ###
print(data[['caste', 'iq']].corr())


#%% 

### PROBLEM TWO ###
import pingouin as pg
pg.partial_corr(data=data, x='caste', y='iq', covar='brainmass')


#%%
### PROBLEM THREE ###
print(data[['iq', 'brainmass']].corr())




#%%
### PROBLEM TWO ###
from simple_linear_regress_func import simple_linear_regress_func # import
from sklearn import linear_model # library for multiple linear regression
import pingouin as pg

X = data[['caste', 'brainmass']]
Y = data[['iq']]

regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
rSqr = regr.score(X,Y) # 0.48 - realistic - life is quite idiosyncratic
betas = regr.coef_ # m
yInt = regr.intercept_  # b




# %%

### QUESTION FOUR ###
from sklearn import linear_model # library for multiple linear regression

X = data[['brainmass']]
Y = data[['iq']]

regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 

print(regr.predict([[3000]]))

#%%

### QUESTION FIVE ###

print(data[['income', 'caste']].corr())


#%%
### QUESTION SIX ###

pg.partial_corr(data=data, x='caste', y='income', covar=['iq', 'hours'])


# %%
### QUESTION SEVEN ###
print(data[['income', 'caste']].corr())

# %%
### QUESTION EIGHT ###
print(data[['income', 'hours']].corr())


#%% 
### QUESTION NINE ###
from sklearn import linear_model # library for multiple linear regression

X = data[['iq', 'hours']]
Y = data[['income']]

regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
regr.score(X,Y)



# %%

### QUESTION TEN ###
from sklearn import linear_model # library for multiple linear regression

X = data[['iq', 'hours']]
Y = data[['income']]

regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 

print(regr.predict([[120, 50]]))
# %%
