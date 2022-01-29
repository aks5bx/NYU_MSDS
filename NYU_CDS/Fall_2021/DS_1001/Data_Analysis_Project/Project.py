#%%

#%%
### SET UP ###
import pandas as pd
import numpy as np
import seaborn as sns 
from scipy import stats
pd.set_option('display.float_format', lambda x: '%.3f' % x)

movie_data = pd.read_csv('movieReplicationSet.csv')
movie_data.head()

#%%
### PROBLEM ONE ###
## TEST: MANN WHITNEY U
# Subset Data Frame to just movies and their user ratings
movie_titles = movie_data.iloc[:,:400]
movie_titles.head()

# Get the median number of ratings for a movie
median_ratings_total = np.median(movie_titles.count().values)

# Get the number of ratings for each movie
movie_titles_numratings = movie_titles.count().to_frame().reset_index()
movie_titles_numratings.columns = ['Movie_Title', 'Num_Ratings']

# Determine hig and low popular movies 
# Note: if a movie's number of ratings was equal to the median, we throw that data out because 
# an perfectly average movie does not help us distinguish between high and low popularity movies
high_popular_movies = list(movie_titles_numratings[movie_titles_numratings.Num_Ratings > median_ratings_total]['Movie_Title'])
low_popular_movies = list(movie_titles_numratings[movie_titles_numratings.Num_Ratings < median_ratings_total]['Movie_Title'])

# Subset data to high and low popular movies
high_popular_movie_titles = movie_titles.loc[:, (movie_titles.columns.isin(high_popular_movies))]
low_popular_movie_titles = movie_titles.loc[:, (movie_titles.columns.isin(low_popular_movies))]

# Get non-nan ratings data for both subsets
high_pop_ratings = high_popular_movie_titles.median().values
low_pop_ratings = low_popular_movie_titles.median().values

# Get p-value of mann whitney u test
t, p = stats.mannwhitneyu(high_pop_ratings, low_pop_ratings, alternative = 'greater')
p

## Final Answer: We reject the null hypothesis that the rating of high-popularity movies
##               is not greater than the rating of low-popularity movies 

# %%
### PROBLEM TWO ###
## TEST: KS Test + MWU
# Initialize movie titles into a list 
movie_titles_list = movie_titles.columns

# Loop through all movie titles, extract title and year
movie_title_year_dict = {}
movie_title_years = []
for title in movie_titles_list:
    year = title.split('(')[-1].split(')')[0]

    try:
        year = int(year)
        movie_title_years.append(year)
        movie_title_year_dict[title] = int(year)
    except:
        print(title)
    
# Save median movie year
movie_title_years = np.array(movie_title_years)
median_movie_year = np.median(movie_title_years)

# Get titles above and below median year
new_movie_titles = [k for k in movie_title_year_dict if movie_title_year_dict[k] > median_movie_year]
old_movie_titles = [k for k in movie_title_year_dict if movie_title_year_dict[k] < median_movie_year]

# Subset data to new and old movies
new_movies = movie_titles.loc[:, (movie_titles.columns.isin(new_movie_titles))]
old_movies = movie_titles.loc[:, (movie_titles.columns.isin(old_movie_titles))]

# Get non-nan ratings data for both subsets
new_movie_ratings = new_movies.median().values
old_movie_ratings = old_movies.median().values

# Plot ratings
new_movie_plot = sns.displot(new_movie_ratings, legend = False)
old_movie_plot = sns.displot(old_movie_ratings, legend = False)

new_movie_plot.fig.suptitle("New Movie Ratings")
old_movie_plot.fig.suptitle("Old Movie Ratings")

t, p = stats.mannwhitneyu(new_movie_ratings, old_movie_ratings, alternative = 'two-sided')
print('Mann Whitney U :', p)

# %%
### PROBLEM THREE ###
## TEST: KS Test + MWU
# Initialize Shrek Data
shrek_data = movie_data[['Shrek (2001)', 'Gender identity (1 = female; 2 = male; 3 = self-described)']]
shrek_data.columns = ['Rating', 'Gender']
# Note: (1 = female; 2 = male; 3 = self-described)

# Subset data on males and females
shrek_males = shrek_data[shrek_data.Gender == 2]
shrek_females = shrek_data[shrek_data.Gender == 1]

shrek_females = shrek_females.dropna()
shrek_males = shrek_males.dropna()

t, p = stats.mannwhitneyu(shrek_males['Rating'], shrek_females['Rating'], alternative = 'two-sided')
print('Mann Whitney U :', p)
# %%
### PROBLEM FOUR ###
## TEST: KS Test
# Functionalize Problem Three
def male_female_difference(movieName):
    specified_movie_data = movie_data[[movieName, 'Gender identity (1 = female; 2 = male; 3 = self-described)']]
    specified_movie_data.columns = ['Rating', 'Gender']

    specified_movie_males = specified_movie_data[specified_movie_data.Gender == 2]
    specified_movie_females = specified_movie_data[specified_movie_data.Gender == 1]

    ## MWU P
    specified_movie_females = specified_movie_females.dropna()
    specified_movie_males = specified_movie_males.dropna()

    ## KS P
    t, p1 = stats.ks_2samp(specified_movie_males['Rating'], specified_movie_females['Rating'], alternative = 'two-sided')

    t, p2 = stats.mannwhitneyu(specified_movie_males['Rating'], specified_movie_females['Rating'], alternative = 'two-sided')

    if p1 <= 0.005 or p2 <= 0.005:
        return 1
    else:
        return 0

significant_gender_results = 0
for specified_movie_title in movie_titles.columns:
    significant_gender_results += male_female_difference(specified_movie_title)


print(significant_gender_results / len(movie_titles.columns))

# %%
### PROBLEM FIVE ###
## TEST: Mann Whitney U
# Initialize Lion King Data 
lion_data = movie_data[['The Lion King (1994)', 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)']]
lion_data.columns = ['Rating', 'Only_Child']

# Note: Only Child? (1: Yes; 0: No; -1: Did not respond)

lion_only = lion_data[lion_data.Only_Child == 1].dropna()
lion_sibling = lion_data[lion_data.Only_Child == 0].dropna()

t, p = stats.mannwhitneyu(lion_only['Rating'], lion_sibling['Rating'], alternative = 'greater')
p
# %%
### PROBLEM SIX ###
## Test: KS Test

# Functionalize Problem 5
def only_child_effect(specified_movie_name):
    spec_movie_data = movie_data[[specified_movie_name, 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)']]
    spec_movie_data.columns = ['Rating', 'Only_Child']

    # Note: Only Child? (1: Yes; 0: No; -1: Did not respond)

    spec_only = spec_movie_data[spec_movie_data.Only_Child == 1].dropna()
    spec_sibling = spec_movie_data[spec_movie_data.Only_Child == 0].dropna()

    t, p = stats.mannwhitneyu(spec_only['Rating'], spec_sibling['Rating'], alternative = 'two-sided')

    if p <= 0.005:
        return 1
    else:
        return 0


only_child_results = 0
for specified_movie_title in movie_titles.columns:
    only_child_results += only_child_effect(specified_movie_title)

print(only_child_results / len(movie_titles.columns))

## TO DO: MWU --> 0.02

# %%
### PROBLEM SEVEN ### 
## TEST: Mann Whitney U 

# Initialize Lion King Data 
wolf_data = movie_data[['The Wolf of Wall Street (2013)', 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)']]
wolf_data.columns = ['Rating', 'Loner']

# Note: Loner? (1: Yes; 0: No; -1: Did not respond)

wolf_loner = wolf_data[wolf_data.Loner == 1].dropna()
wolf_social = wolf_data[wolf_data.Loner == 0].dropna()

t, p = stats.mannwhitneyu(wolf_social['Rating'], wolf_loner['Rating'], alternative = 'greater')
p

# %%
### PROBLEM EIGHT ###
## TEST: KS Test

# Functionalize Problem 7
def social_effect(specified_movie_name):
    spec_movie_data = movie_data[[specified_movie_name, 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)']]
    spec_movie_data.columns = ['Rating', 'Loner']

    # Note: Loner? (1: Yes; 0: No; -1: Did not respond)

    spec_loner = spec_movie_data[spec_movie_data.Loner == 1].dropna()
    spec_social = spec_movie_data[spec_movie_data.Loner == 0].dropna()

    t, p = stats.mannwhitneyu(spec_loner['Rating'], spec_social['Rating'], alternative = 'two-sided')

    if p <= 0.005:
        return 1
    else:
        return 0


social_results = 0
for specified_movie_title in movie_titles.columns:
    social_results += social_effect(specified_movie_title)

print(social_results / len(movie_titles.columns))

## TO DO: Add MWU --> 0.025
# %%
### PROBLEM NINE ###
## TEST: KS Test
t, p = stats.ks_2samp(movie_data['Home Alone (1990)'].dropna(), movie_data['Finding Nemo (2003)'].dropna(), alternative = 'two-sided')
p

# %%
### PROBLEM TEN ###
## Test: Kruskal Wallis 

franchises = ['Star Wars', 'Harry Potter', 'The Matrix', 'Indiana Jones', 
            'Jurassic Park', 'Pirates of the Caribbean', 'Toy Story', 'Batman']

for franchise in franchises: 
    movies = []
    for movie_title in list(movie_titles.columns):
        if franchise in movie_title:
            movies.append(movie_title)
    print(franchise)
    print(movies)
    #print('----------')

#%%
# Star Wars
star_wars = movie_titles[['Star Wars: Episode IV - A New Hope (1977)', 'Star Wars: Episode II - Attack of the Clones (2002)', 'Star Wars: Episode V - The Empire Strikes Back (1980)', 'Star Wars: Episode 1 - The Phantom Menace (1999)', 'Star Wars: Episode VII - The Force Awakens (2015)', 'Star Wars: Episode VI - The Return of the Jedi (1983)']].dropna()
h, p = stats.kruskal(star_wars['Star Wars: Episode IV - A New Hope (1977)'], 
                    star_wars['Star Wars: Episode II - Attack of the Clones (2002)'], 
                    star_wars['Star Wars: Episode V - The Empire Strikes Back (1980)'], 
                    star_wars['Star Wars: Episode 1 - The Phantom Menace (1999)'], 
                    star_wars['Star Wars: Episode VII - The Force Awakens (2015)'], 
                    star_wars['Star Wars: Episode VI - The Return of the Jedi (1983)'],
                    nan_policy = 'omit')

print('Star Wars', p)

# Harry Potter
harry_potter = movie_titles[["Harry Potter and the Sorcerer's Stone (2001)", 'Harry Potter and the Deathly Hallows: Part 2 (2011)', 'Harry Potter and the Goblet of Fire (2005)', 'Harry Potter and the Chamber of Secrets (2002)']].dropna()
h, p = stats.kruskal(harry_potter["Harry Potter and the Sorcerer's Stone (2001)"], 
                    harry_potter['Harry Potter and the Deathly Hallows: Part 2 (2011)'], 
                    harry_potter['Harry Potter and the Goblet of Fire (2005)'], 
                    harry_potter['Harry Potter and the Chamber of Secrets (2002)'], 
                    nan_policy = 'omit')

print('Harry Potter', p)


# The Matrix
matrix = movie_titles[['The Matrix Revolutions (2003)', 'The Matrix Reloaded (2003)', 'The Matrix (1999)']].dropna()
h, p = stats.kruskal(matrix["The Matrix Revolutions (2003)"], 
                    matrix['The Matrix Reloaded (2003)'], 
                    matrix['The Matrix (1999)'], 
                    nan_policy = 'omit')

print('The Matrix', p)

# Indiana Jones
indiana_jones = movie_titles[['Indiana Jones and the Last Crusade (1989)', 'Indiana Jones and the Temple of Doom (1984)', 'Indiana Jones and the Raiders of the Lost Ark (1981)', 'Indiana Jones and the Kingdom of the Crystal Skull (2008)']].dropna()
h, p = stats.kruskal(indiana_jones["Indiana Jones and the Last Crusade (1989)"], 
                    indiana_jones['Indiana Jones and the Temple of Doom (1984)'], 
                    indiana_jones['Indiana Jones and the Raiders of the Lost Ark (1981)'], 
                    indiana_jones['Indiana Jones and the Kingdom of the Crystal Skull (2008)'],
                    nan_policy = 'omit')

print('Indiana Jones', p)


# Jurassic Park
j_park = movie_titles[['The Lost World: Jurassic Park (1997)', 'Jurassic Park III (2001)', 'Jurassic Park (1993)']].dropna()
h, p = stats.kruskal(j_park["The Lost World: Jurassic Park (1997)"], 
                    j_park['Jurassic Park III (2001)'], 
                    j_park['Jurassic Park (1993)'], 
                    nan_policy = 'omit')

print('Jurassic Park', p)


# Pirates
pirates = movie_titles[["Pirates of the Caribbean: Dead Man's Chest (2006)", "Pirates of the Caribbean: At World's End (2007)", 'Pirates of the Caribbean: The Curse of the Black Pearl (2003)']].dropna()
h, p = stats.kruskal(pirates["Pirates of the Caribbean: Dead Man's Chest (2006)"], 
                    pirates["Pirates of the Caribbean: At World's End (2007)"], 
                    pirates['Pirates of the Caribbean: The Curse of the Black Pearl (2003)'],
                    nan_policy = 'omit')

print('Pirates', p)


# Toy Story
toy_story = movie_titles[['Toy Story 2 (1999)', 'Toy Story 3 (2010)', 'Toy Story (1995)']].dropna()
h, p = stats.kruskal(toy_story["Toy Story 2 (1999)"], 
                    toy_story['Toy Story 3 (2010)'], 
                    toy_story['Toy Story (1995)'],
                    nan_policy = 'omit')

print('Toy Story', p)


# Batman
batman = movie_titles[['Batman & Robin (1997)', 'Batman (1989)', 'Batman: The Dark Knight (2008)']].dropna()
h, p = stats.kruskal(batman["Batman & Robin (1997)"], 
                    batman['Batman (1989)'], 
                    batman['Batman: The Dark Knight (2008)'],
                    nan_policy = 'omit')

print('Batman', p)

# %%
