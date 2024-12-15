import pandas as pd
df = pd.read_csv("price2.csv")
#print(df.head)
#print(pd.get_dummies(df.town))
dummies = pd.get_dummies(df.town)

merged = pd.concat([df,dummies], axis='columns')
#print(merged)

final = merged.drop(['town','west windsor'],axis='columns')
#print(final)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

X = final.drop('price',axis='columns')
#print(X)
Y= final.price
#print(Y)

model.fit(X,Y)
#print(model.predict([[2800,0,1]]))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfle = df
dfle.town = le.fit_transform(dfle.town)
print(dfle)

X1 = dfle[['town','area']].values
print(X1)

Y1 = dfle.price
print(Y1)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
print(ohe.fit_transform(X1).toarray())

X1 = X1[:,1:]
print(X1)


model.fit(X1,Y1)
R = model.predict([[1,0,2800]]) 
print(R)
