#%%
import pandas as pd
data = pd.read_csv('Invisibility Cloak.csv')
from scipy import stats
#%%

### PROBLEM 41 ###
no_cloak = data[data.Cloak == 0]
cloak = data[data.Cloak == 1]

t, p = stats.mannwhitneyu(no_cloak.Mischief, cloak.Mischief, alternative = 'two-sided')
p

stats.ttest_ind(no_cloak.Mischief, cloak.Mischief)
# %%
### PROBLEM 42 ###

data.head(25)
# %%
data = pd.read_csv('DarkTriad.csv')

#%%
### PROBLEM 45 ###

data[['Narcissism', 'Machiavellianism']].corr()
# %%
### PROBLEM 46 ###

psych = data[data.Psychopathy > 30]

len(psych) / len(data)
# %%
data = pd.read_csv('moonAndAggression.txt', sep='\t')
# %%
data.head()
# %%
### PROBLEM 47 ###
len(data)
# %%

data = pd.read_csv('thanks.csv')
# %%
### PROBLEM 49 ### 

len(data)
# %%
data = pd.read_csv('Sadex2.csv')
# %%
### PROBLEM 50 ###

stats.ttest_rel(data.Before, data.After)
# %%
