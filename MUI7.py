import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("insur7.csv")
insurance = df.head()
print(insurance)

plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.1)
print(X_test)   

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)

pred1 = model.predict(X_test)
print(pred1)
score1 = model.score(X_test,Y_test)
print(score1)

prob1 = model.predict_proba(X_test)
print(prob1)