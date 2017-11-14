''' This script impliments an ensemble model '''

# Basics 
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.metrics import r2_score as r2


# Machine learning  
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit

# Plotting 
import seaborn as sb
import matplotlib as matplot
import matplotlib.pyplot as plot


# Import data
data = pd.read_csv('./Data/data.csv')
x = data.drop('price', axis = 1)
y = data.price
x = x.as_matrix()
y = y.as_matrix()

# Split into simple training and test sets
Xtrain = x[0:18000, :]
Ytrain = y[0:18000]
Xtest = x[18001:len(x), :]
Ytest = y[18001:len(y)]

# Because this is time series data, I use TimeSeriesSplit to obtain indices of training and CV sets (k-fold for time series)
tscv = TimeSeriesSplit(n_splits = 10)

#%%

# Creating model objects and list that will contain info from each CV fold
SVR = svm.SVR()
SVRinfo = []

RFR = RandomForestRegressor(max_depth = 4, random_state=0)
RFRinfo = []

BR = BayesianRidge()
BRinfo = []

NN = MLPRegressor()
NNinfo = []

ADA = AdaBoostRegressor()
ADAinfo = []


# Training and displaying the R2 and squared error for each iteration of K-fold
iteration = 1
for train_index, test_index in tscv.split(Ytrain):
    print('CV Fold %d' %(iteration))
    
    '''SVR'''
    # Train the model
    SVR.fit(Xtrain[train_index, :], Ytrain[train_index])
    
    # Calculate R2 scores for training and CV set
    train_score = SVR.score(Xtrain[train_index], Ytrain[train_index])
    test_score = SVR.score(Xtrain[test_index], Ytrain[test_index])
    
    # Calculate the squared error for training and CV set, save in list
    train_error = (1/len(train_index))*sum((SVR.predict(Xtrain[train_index]) - Ytrain[train_index])**2)
    test_error = (1/len(test_index))*sum((SVR.predict(Xtrain[test_index]) - Ytrain[test_index])**2)
    SVRinfo.append('**SVR**    R2 in I%d: Train: %f, Test: %f' %(iteration, train_score, test_score))
    SVRinfo.append('**SVR** Error in I%d: Train: %f, Test: %f' %(iteration, train_error, test_error))



    '''Random Forest Regressor'''
    # Train the model
    RFR.fit(Xtrain[train_index, :], Ytrain[train_index])
    
    # Calculate R2 scores for training and CV set
    train_score = RFR.score(Xtrain[train_index], Ytrain[train_index])
    test_score = RFR.score(Xtrain[test_index], Ytrain[test_index])
    
    # Calculate the squared error for training and CV set, save in list
    train_error = (1/len(train_index))*sum((RFR.predict(Xtrain[train_index]) - Ytrain[train_index])**2)
    test_error = (1/len(test_index))*sum((RFR.predict(Xtrain[test_index]) - Ytrain[test_index])**2)
    RFRinfo.append('**RFR**    R2 in I%d: Train: %f, Test: %f' %(iteration, train_score, test_score))
    RFRinfo.append('**RFR** Error in I%d: Train: %f, Test: %f' %(iteration, train_error, test_error))
    
    
    
    '''Bayesian Ridge'''
    # Train the model
    BR.fit(Xtrain[train_index, :], Ytrain[train_index])
    
    # Calculate R2 scores for training and CV set
    train_score = BR.score(Xtrain[train_index], Ytrain[train_index])
    test_score = BR.score(Xtrain[test_index], Ytrain[test_index])
    
    # Calculate the squared error for training and CV set, save in list
    train_error = (1/len(train_index))*sum((BR.predict(Xtrain[train_index]) - Ytrain[train_index])**2)
    test_error = (1/len(test_index))*sum((BR.predict(Xtrain[test_index]) - Ytrain[test_index])**2)
    BRinfo.append('**BR**    R2 in I%d: Train: %f, Test: %f' %(iteration, train_score, test_score))
    BRinfo.append('**BR** Error in I%d: Train: %f, Test: %f' %(iteration, train_error, test_error))
    
    
    
    '''Neural Network Regressor'''
    # Train the model
    NN.fit(Xtrain[train_index, :], Ytrain[train_index])
    
    # Calculate R2 scores for training and CV set
    train_score = NN.score(Xtrain[train_index], Ytrain[train_index])
    test_score = NN.score(Xtrain[test_index], Ytrain[test_index])
    
    # Calculate the squared error for training and CV set, save in list
    train_error = (1/len(train_index))*sum((NN.predict(Xtrain[train_index]) - Ytrain[train_index])**2)
    test_error = (1/len(test_index))*sum((NN.predict(Xtrain[test_index]) - Ytrain[test_index])**2)
    NNinfo.append('**NN**    R2 in I%d: Train: %f, Test: %f' %(iteration, train_score, test_score))
    NNinfo.append('**NN** Error in I%d: Train: %f, Test: %f' %(iteration, train_error, test_error))
    
    
    
    '''AdaBoost Regressor'''
    # Train the model
    ADA.fit(Xtrain[train_index, :], Ytrain[train_index])
    
    # Calculate R2 scores for training and CV set
    train_score = ADA.score(Xtrain[train_index], Ytrain[train_index])
    test_score = ADA.score(Xtrain[test_index], Ytrain[test_index])
    
    # Calculate the squared error for training and CV set, save in list
    train_error = (1/len(train_index))*sum((ADA.predict(Xtrain[train_index]) - Ytrain[train_index])**2)
    test_error = (1/len(test_index))*sum((ADA.predict(Xtrain[test_index]) - Ytrain[test_index])**2)
    ADAinfo.append('**ADA**    R2 in I%d: Train: %f, Test: %f' %(iteration, train_score, test_score))
    ADAinfo.append('**ADA** Error in I%d: Train: %f, Test: %f' %(iteration, train_error, test_error))
    
    
    
    iteration += 1

# Printing training info
print(*SVRinfo, sep='\n')
print('\n**********************')
print(*RFRinfo, sep='\n')
print('\n**********************')
print(*BRinfo, sep='\n')
print('\n**********************')
print(*NNinfo, sep='\n')
print('**********************\n')
print(*ADAinfo, sep='\n')
print('**********************\n')


# Printing score on test set
print('SVR test score: %f' %(SVR.score(Xtest, Ytest)))
print('RFR test score: %f' %(RFR.score(Xtest, Ytest)))
print('BR test score: %f' %(BR.score(Xtest, Ytest)))
print('NN test score: %f' %(NN.score(Xtest, Ytest)))
print('ADA test score: %f' %(ADA.score(Xtest, Ytest)))


#%% 
'''Simple Stacking: Averaging results'''

# Defining 
#def r2(prediction, actual):
#    u = np.sum((actual - prediction)**2)
#    v = np.sum((actual - actual.mean())**2)
#    uv = u/v
#    return 1 - uv

# Calcuating predictions with test set
SVRstack = SVR.predict(Xtest)
RFRstack = RFR.predict(Xtest)
BRstack = BR.predict(Xtest)
NNstack = NN.predict(Xtest)
ADAstack = ADA.predict(Xtest)

# Calculating mean of predictions
number_of_estimators = 5
meanPrediction = (SVRstack + RFRstack + BRstack + NNstack + ADAstack)/number_of_estimators

print('Simple Average Ensemble Score: %f' %(r2(meanPrediction, Ytest)))

# Saving predictions to CSV
SVRstack.tofile('SVR_prediction.csv', sep = ',', format = '%f')
RFRstack.tofile('RFR_prediction.csv', sep = ',', format = '%f')
BRstack.tofile('BR_prediction.csv', sep = ',', format = '%f')
NNstack.tofile('NN_prediction.csv', sep = ',', format = '%f')
ADAstack.tofile('ADA_prediction.csv', sep = ',', format = '%f')
meanPrediction.tofile('mean_prediction.csv', sep = ',', format = '%f')

#%%
#def reverse_transform(y):
#    return np.exp(y)+1
#
#Yactual = reverse_transform(Ytest)
#SVRactual = reverse_transform(svr.predict(Xtest))
#RFRactual = reverse_transform(RFR.predict(Xtest))
#BRactual = reverse_transform(BR.predict(Xtest))
#NNactual = reverse_transform(NN.predict(Xtest))
#ADAactual = reverse_transform(ADA.predict(Xtest))
#
#SVRpdiff = (Yactual-SVRactual)/Yactual*100
#SVRpdiff_mean = (1/len(Ytest))*sum(SVRpdiff)