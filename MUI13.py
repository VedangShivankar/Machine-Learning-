import pandas as pd
df = pd.read_csv("oof13.csv")
print(df.head)
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head)
target = df.Survived
inputs= df.drop('Survived', axis='columns')
dummies= pd.get_dummies(inputs.Sex)
gender = dummies.head = 3
inputs.columns[inputs.isna().any()]
inputs.Age[:10]

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(inputs,test_size=0.2)
print(len(X_train))
print(len(X_test))
print(len(inputs))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
ddddd = model.score(X_train,y_train)