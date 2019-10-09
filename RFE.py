#Recursive Feature Elimination

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)

model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
X_new = rfe.fit_transform(dataframe.iloc[:,0:8], dataframe.iloc[:,8])

X_new.n_features_ # returns the number of features
X_new.support_ # returns the selected features
X_new.ranking_ # returns the ranking