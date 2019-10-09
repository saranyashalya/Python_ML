from sklearn.decomposition import PCA
import pandas as pd

#load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)

pca = PCA(n_components=3)
dataframe_fit = pca.fit(dataframe.iloc[:,0:8], dataframe.iloc[:8])

dataframe_fit.explained_variance_ratio_
dataframe_fit.components_

dataframe_trans = pca.fit_transform(dataframe.iloc[:,0:8], dataframe.iloc[:8])
print(dataframe_trans[0:5,:])
