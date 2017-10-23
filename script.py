     
import pandas as pd
import sklearn as sk
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import seaborn as sb
import matplotlib as matplot
import matplotlib.pyplot as plot
import copy

#%% 
''' Data importing and feature engineering '''

# Load the data into a Pandas dataframe
data = pd.read_csv('kc_house_data.csv')

# Sort according to date
data = data.sort_values('date')

# Creating new index, ordered by date
data = data.reset_index(drop=True)

# Define target variable, drop features that won't be used
y = data.price
compdata = data.drop(['id', 'date'], axis = 1)

## The magnitudes of most of the features are very different from one another, so normalize right away
#for column in compdata:
#    compdata[column] = (compdata[column] - compdata[column].mean())/compdata[column].std()

# Checking for null values:
nulls = sum(data.isnull().sum()) # No missing values!

 
#%% Plotting for an overview of the data: distribution and correlation plots to check for skew and multicollinearity 

# First, a correlation matrix
corr = compdata.corr()
sb.heatmap(corr, annot=True)
plot.xticks(rotation = 45)
plot.yticks(rotation = 0)

#%%
'''
It looks like the dataset is fairly well behaved in terms of correlation between the independent variables, but there are some features 
that are very poorly correlated to the sale price. Some of these poorly correlated features are surprising, like zipcode for example.
Lets take a closer look at them vs. price
'''

investigate = ['sqft_lot', 'condition', 'yr_built', 'zipcode', 'long']

# Plot the features in question vs. price
f1, ax1 = plot.subplots(2,3)
ax1 = ax1.ravel()

for i in range(len(investigate)):
    ax1[i].scatter(data[investigate[i]], y, s=1, c='k')
    ax1[i].set_title(investigate[i])

#%%
''' Engineering some features '''

# Instead of using the actual zipcode as a feature, I'll use the average price of the zipcode from the entire dataset
tempzip = data['zipcode']
 
# Making an array with all unique zipcodes:
zips, inverse = np.unique(tempzip, return_inverse = True)

# Making an array with the mean price per zipcode 
zipmeans = np.ones(len(zips))
for i in range(len(zips)):
    tempy = copy.copy(y)
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
ax2.scatter(compdata.meanzip, y, s=1, c='k')
ax2.axes.set_title('meanzip')

# For now, drop the other poorly correlated features
drop = ['sqft_lot', 'condition', 'yr_built', 'long']
compdata = compdata.drop(drop, axis=1)

# Make a new correlation heatmap to check on things
corr = compdata.corr()
sb.heatmap(corr, annot = True)
plot.xticks(rotation = 45)
plot.yticks(rotation = 0)


#%% Creating a distribution plot

# Making a list of features to label the plot with, and a matrix from compdata for plotting
features = list(compdata.axes[1])
plotdata = np.asmatrix(compdata)

# Making the distribution plot
f, ax  = plot.subplots(3,5)
ax = ax.ravel()

for i in range(len(features)):
    sb.distplot(plotdata[:,i], ax = ax[i])
    ax[i].set_title(features[i])

# Looks like 

    
#
## Because this is time series data, I use TimeSeriesSplit to obtain indices of training and CV sets (k-fold for time series)
#tscv = TimeSeriesSplit(n_splits = 10)
#
#for train_index, test_index in tscv.split(data):
#    print('Train:', train_index, 'Test:', test_index)





