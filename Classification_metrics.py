from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold

#load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)

seed = 7
test_size = 0.3
X = dataframe.iloc[:,0:8]
Y = dataframe.iloc[:,8]
kfold = KFold(n_splits=10, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
## cross validation accuracy
print('Accuracy %.3f%% (std-%.3f%%)' %(results.mean()*100.0, results.std()*100.0))

##Logarithmic loss
scoring ='neg_log_loss'
results_log = cross_val_score(model, X, Y, scoring = scoring, cv = kfold)
print("Log loss : %.3f%% (std = %.3f%%)" %(results_log.mean()*100.0, results_log.std()*100.0))

##Area under curve
scoring='roc_auc'
results_auc = cross_val_score(model, X, Y, scoring= scoring, cv = kfold)
print("auc %.3f%% (std - %.3f%%)" %(results_auc.mean()*100.0, results_auc.std()*100.0))

## Confusion matrix
X_train,X_test, Y_train, Y_test = train_test_split(X,Y, random_state=seed, test_size= test_size)
model.fit(X_train, Y_train)
preds = model.predict(X_test)
print(confusion_matrix(Y_test, preds))

## Classification report
print(classification_report(Y_test, preds))
