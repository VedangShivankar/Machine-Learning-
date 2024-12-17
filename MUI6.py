import pandas as pd
df = pd.read_csv('cars6.csv')
print(df.head())

import matplotlib.pyplot as plt
%mathplotlib inline
print(plt.scatter(df['Mileage'], df['Sell Price($)']))
print(plt.scatter(df['Age(yrs)'], df['Sell Price($)']))

X = df[['Mileage','Age(yrs)']]
Y = df['Sell Price($)']
print(Y)
print(X)

from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
print(len(X_train))

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,Y_train)
print(clf.predict(X_test))
print(clf.predict(Y_test))
accurate = clf.score(X_test,Y_test)
print(accurate)