     
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plot
import copy

#%% 
''' Data importing and feature engineering '''

# Load the data into a Pandas dataframe
data = pd.read_csv('./Data/kc_house_data.csv')

# Sort according to date
data = data.sort_values('date')

# Creating new index, ordered by date
data = data.reset_index(drop=True)

# Drop features that won't be used
compdata = data.drop(['id', 'date', 'long'], axis = 1) # 'long' is just one value all throughout the dataset, so it's useless in this model

# Checking for null values:
nulls = sum(data.isnull().sum()) # No missing values!

# Checking for negative values:
for column in compdata:
    if np.sum(compdata[column] < 0) > 0:
        print('Negative values exist in %s' %(column))
# No negative values!

 
#%% 
''' Plotting for an overview of the data: distribution and correlation plots to check for skew and multicollinearity '''

# First, a correlation matrix
corr = compdata.corr()
sb.heatmap(corr, annot=True)
plot.xticks(rotation = 45)
plot.yticks(rotation = 0)

#It looks like the dataset is fairly well behaved in terms of multicorrilarity, but it looks like 'sqft_above' and 'sqft_living' 
#are so closely correlated that it's probably best to drop one of them. I'll arbitrarily choose 'sqft_above' to drop.
compdata = compdata.drop('sqft_above', axis = 1)

#investigate = ['sqft_lot', 'condition', 'yr_built', 'zipcode', 'long']
#
## Plot the features in question vs. price
#f1, ax1 = plot.subplots(2,3)
#ax1 = ax1.ravel()
#
#for i in range(len(investigate)):
#    ax1[i].scatter(data[investigate[i]], y, s=1, c='k')
#    ax1[i].set_title(investigate[i])

#%%
''' Engineering some features '''

# Instead of using the actual zipcode as a feature, I'll use the average price of the zipcode from the entire dataset
tempzip = data['zipcode']
 
# Making an array with all unique zipcodes:
zips, inverse = np.unique(tempzip, return_inverse = True)

# Making an array with the mean price per zipcode 
zipmeans = np.ones(len(zips))
for i in range(len(zips)):
    tempy = copy.copy(compdata.price)
    zipmask = 1*(tempzip == zips[i])
    tempy *= zipmask
    zipmeans[i] = tempy[tempy!=0].mean()

# Creating an array with the new values
tempzip = np.ones(len(inverse))
for i in range(len(tempzip)):
    tempzip[i] = zipmeans[inverse[i]]    
compdata.zipcode = tempzip

# Renaming zipcode feature
compdata = compdata.rename(columns = {'zipcode':'meanzip'})    

# Make a plot of the new feature
f2 = plot.figure() 
ax2 = f2.add_subplot(111)
ax2.scatter(compdata.meanzip, compdata.price, s=1, c='k')
ax2.axes.set_title('meanzip')

# For now, drop the other poorly correlated features
#drop = ['sqft_lot', 'condition', 'yr_built', 'long']
#compdata = compdata.drop(drop, axis=1)

# Make a new correlation heatmap to check on things
corr = compdata.corr()
sb.heatmap(corr, annot = True)
plot.xticks(rotation = 45)
plot.yticks(rotation = 0)

#%% 

''' Features are selected and ready for the next step of preprocessing. '''

# Features are very different in terms of scale, so time for feature scaling. I'm choosing to do mean-normalization.
#for column in compdata:
#    compdata[column] = (compdata[column] - compdata[column].min())/(compdata[column].max() - compdata[column].min())
#

# Making a list of features to label the plot with, and a matrix from compdata for plotting
features = list(compdata.axes[1])
plotdata = np.asmatrix(compdata)

# Making the distribution plot
f3, ax3  = plot.subplots(3,6)
ax3 = ax3.ravel()

for i in range(len(features)):
    sb.distplot(plotdata[:,i], ax = ax3[i])
    ax3[i].set_title(features[i])

# There is significant right skew in most of the variables, and left skew in a couple, so it's probably worth normalizing them.
right_normalize = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_basement', 'meanzip', 'sqft_living15', 'floors']
left_normalize = ['lat', 'yr_built', ]

# log+1 transformation to correct for right skew
for feature in right_normalize:
    compdata[feature] = np.log1p(compdata[feature])

# Power transformations to correct for left skew
compdata.lat = compdata.lat**7
compdata.yr_built = compdata.yr_built**5

# Plot again to visualize transformation
plotdata = np.asmatrix(compdata)
f4, ax4  = plot.subplots(3,6)
ax4 = ax4.ravel()

for i in range(len(features)):
    sb.distplot(plotdata[:,i], ax = ax4[i])
    ax4[i].set_title(features[i])
# Features look much better.

# Features are very different in terms of scale, so time for feature scaling. I'm choosing to do mean-normalization.
for column in compdata.columns:  #.drop('price', axis = 1):
    compdata[column] = (compdata[column] - compdata[column].min())/(compdata[column].max() - compdata[column].min())
#%% 
''' I think the data is ready for a preliminary model. Time to split the data into training and CV sets '''    

# Saving to CSV
compdata.to_csv('./Data/data.csv', index = False)






