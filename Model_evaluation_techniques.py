from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit, cross_val_score, train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression

#load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names = names)

X = dataframe.iloc[:,0:8]
Y = dataframe.iloc[:,8]
# method 1 - Train test split
seed = 7
test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state= seed)

# model building
model = LogisticRegression()
model.fit(X_train, Y_train)
results = model.score(X_test, Y_test)
print("Accuracy(train,test split) : %.3f%%" %(results*100.00))

## Kfold cross validation
kfold = KFold(n_splits=10, random_state=seed)
results_kfold = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy(Kfold) : %.3f%% (std - %.3f)" %(results_kfold.mean()*100.0, results_kfold.std()*100.0))

## Leave one out cross validation
loocv = LeaveOneOut()
results_loocv = cross_val_score(model, X, Y, cv = loocv)
print("Accuracy(LOOCV) : %.3f%% (std - %.3f%%) " %(results_loocv.mean()*100.0, results_loocv.std()*100.0))

## repeated random train test split
kfold = ShuffleSplit(n_splits=10, random_state= seed)
results_shuffle = cross_val_score(model, X, Y, cv= kfold)
print("Accuracy(shuffle) %.3f%% (std - %.3f%%)" %(results_shuffle.mean()*100.0, results_shuffle.std()*100.0))

