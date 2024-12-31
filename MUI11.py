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


plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
f = plt.scatter(df0['sepal length (cm)'],df0['sepal length (cm)'], color='green',marker='+')
g = plt.scatter(df1['sepal length (cm)'],df1['sepal length (cm)'], color='blue',marker='.')


plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
f = plt.scatter(df0['sepal length (cm)'],df0['petal length (cm)'], color='green',marker='+')
g = plt.scatter(df1['sepal length (cm)'],df1['petal length (cm)'], color='blue',marker='.')

from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'],axis='columns')
print(X.head())
y = df.target 

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
len(X_train)
len(X_test)

from sklearn.svm import SVC 
model = SVC()
model.fit(X_train,y_train)
qwe = model.score(X_test,y_test)
print(qwe)