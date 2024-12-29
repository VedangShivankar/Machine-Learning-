import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
dir(iris)
iris.feature_names

df = pd.DataFrame(iris.data, columns=iris.feature_names)
#print(df.head())
df['target'] = iris.target
iris.target_names
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
#print(df.head())

from matplotlib import pyplot as plt
df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]
#print(df2.head())
f = plt.scatter(df0['sepal length (cm)'],df0['sepal length (cm)'], color='green',marker='+')
g = plt.scatter(df1['sepal length (cm)'],df1['sepal length (cm)'], color='blue',marker='+')
print(f)