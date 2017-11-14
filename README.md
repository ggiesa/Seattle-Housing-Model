# Seattle-Housing-Model
This is a relatively simple machine learning project, with the goal of prediciting Seattle housing prices from a 2014-2015 dataset found [here](https://www.kaggle.com/harlfoxem/housesalesprediction/data "King County Housing Data").

I've implemented a simple ensemble regression technique using SVM, Bayesian ridge, random forest, adaboost, and neural network regressors from Scikit-learn. The project is on ice at the moment, after achieving a moderate R2 score of ~80. In terms of improving the model, there are many low hanging fruit, but from my preliminary analysis I've come to suspect that that the dataset lacks the complexity necessary to build a truly useful predictive model. For example, the model achieves a percent error of around 3%, but an absolute percent error of around 20%, indicating to me that there are trends in the target variable that cannot be explained by the rest of the features. 

