import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

#Load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(url, delim_whitespace=True, names=names)

X = dataframe.iloc[:,0:13]
Y = dataframe.iloc[:,13]

seed = 7
kfold = KFold(n_splits=10, random_state=seed)
model = LinearRegression()

##Mean absolute error
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("Mean Absolute error :  %.3f (std - %.3f)" %(results.mean(), results.std()))

## mean squared error
scoring ='neg_mean_squared_error'
results =cross_val_score(model, X, Y, scoring = scoring, cv = kfold)
print("Mean Squared error : %.3f (std:%.3f)" %(results.mean(), results.std()))

##R^2
scoring='r2'
results = cross_val_score(model, X, Y, scoring= scoring, cv=kfold)
print("r2 : %.3f (std : %.3f)" %(results.mean(), results.std()))