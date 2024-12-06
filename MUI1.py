import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model    

df = pd.read_csv("price1.csv")
print(df)

%matplotlib inline
plt.xlabel=('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.price, color="blue")

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
print(reg.predict(3000))
print(reg.coef_)
print(reg.intercept)

d = pd.read_csv("area1.csv")
print(d.head(3))
p = reg.predict(d)
d['price'] = p   
print(d)
d.to_csv("area1.csv")