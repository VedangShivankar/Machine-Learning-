import pandas as pd
import numpy as np
from sklearn import linear_model    

df = pd.read_csv("homeprice2.csv")
print(df.bedrooms.median())

import math
median_bedrooms = math.floor(df.bedrooms.median())
print(median_bedrooms)

df.bedrooms =df.bedrooms.fillna(median_bedrooms)
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[3000,3,40]]))
