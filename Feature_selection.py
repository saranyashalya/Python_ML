from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression

iris = load_iris()
X,Y = iris.data, iris.target
X.shape
Y.shape

X_new = SelectKBest(score_func=chi2, k=2).fit_transform(X,Y)

#For regression: f_regression, mutual_info_regression
#For classification: chi2, f_classif, mutual_info_classif

##regression
boston = load_boston()
X_reg,Y_reg = boston.data, boston.target

X_reg_new = SelectKBest(score_func=f_regression, k= 5).fit_transform(X_reg,Y_reg)

