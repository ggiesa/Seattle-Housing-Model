'''With the from-scratch regression model working and performing somewhat decently, 
   this script will now explore some other regression techniques with other packages
        Models to try: 
            - Ridge regression
            - Random forest regression
            - Lasso regression
            - Bayesian Ridge regression
            - SVM
            - NN
'''        
    
import pandas as pd
import sklearn as sk
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import seaborn as sb
import matplotlib as matplot
import matplotlib.pyplot as plot


# Load the data into a Pandas dataframe
data = pd.read_csv('kc_house_data.csv')

# Define target variable, drop features that won't be used
y = data.price
#Xdata = data.drop(['price', 'id', 'date'], axis = 1)
compdata = data.drop(['id', 'date'], axis = 1)


# The magnitudes of the features are very different from one another, so normalize right away
for column in compdata:
#    Xdata[column] = (Xdata[column] - Xdata[column].mean())/Xdata[column].std()
    compdata[column] = (compdata[column] - compdata[column].mean())/compdata[column].std()

# Checking for null values:
nulls = sum(data.isnull().sum()) # No missing values!

 
#%% Plotting for an overview of the data: distribution and correlation plots to check for skew and multicollinearity 

# Dropping the date feature for visualization
#compdata = data.drop(['date', 'id'],  axis = 1)

# First, a correlation matrix
corr = compdata.corr()
sb.heatmap(corr, annot=True)
plot.xticks(rotation = 45)
plot.yticks(rotation = 0)

'''
It looks like the dataset is fairly well behaved in terms of correlation between the independent variables, but there are some features 
that are very poorly correlated to the sale price. Some of these poorly correlated features are surprising, like zipcode for example.
Lets take a closer look at them vs. price
'''

#sb.pairplot()

drop = ['sqft_lot', 'condition', 'yr_built', 'zipcode', 'long']
compdata = compdata.drop(drop, axis=1)
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





