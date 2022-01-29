#%%
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
plt.close("all")
np.random.seed(2017)

def p_longest_streak(n, tries):
    # Write your Monte Carlo code here, n is the length of the sequence and tries is the number
    # of sampled sequences used to produce the estimate of the probability
    
    maxStreaks = []
    for i in range (0,tries):
        flips = list(itertools.product("HT", repeat=n))
        flip = random.choice(flips)
        
        maxStreakLocal = getMaxStreak(flip, n)
        
        maxStreaks.append(maxStreakLocal)
        
        
    lenMaxStreaks = len(maxStreaks)
    
    probabilities = []
    for i in range(0, n+1):
        prob = maxStreaks.count(i) / lenMaxStreaks
        probabilities.append(prob)
        
        
        
        
        
    return probabilities
        
        


def getMaxStreak(flip_list, n):
    streak = 0
    maxStreak = 0
    iter = 1
    for result in flip_list:
        
        if result == 'H':
            streak += 1

            
        else:

            if streak > maxStreak:
                maxStreak = streak
            streak = 0
            
            
        if iter == n and streak >= maxStreak:
            maxStreak = streak
            
        iter += 1
        
    
    return maxStreak




#%%

x = p_longest_streak(5, 200)

    
#%%
n_tries = [10] ##[1e3,5e3,1e4,5e4,1e5]

n_vals = [200] ##[5,200]

color_array = ['orange','darkorange','tomato','red', 'darkred', 'tomato', 'purple', 'grey', 'deepskyblue', 
               'maroon','darkgray','darkorange', 'steelblue', 'forestgreen', 'silver']
for ind_n in range(len(n_vals)):
    n = n_vals[ind_n]
    plt.figure(figsize=(20,5))
    for ind_tries in range(len(n_tries)):
        tries = n_tries[ind_tries]
        print ("tries: " + str(tries))
        p_longest_tries = p_longest_streak(n, np.int(tries))
        plt.plot(range(n+1),p_longest_tries, marker='o',markersize=6,linestyle="dashed",lw=2,
                 color=color_array[ind_tries],
                 markeredgecolor= color_array[ind_tries],label=str(tries))
    plt.legend()
    
print( "The probability that the longest streak of ones in a Bernoulli iid sequence of length 200 has length 8 or more is ")
print() # Compute the probability and print it here







# %%
