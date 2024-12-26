import pandas as pd
df = pd.read_csv("salary9.csv")
inputs = df.drop('salary_more_then_100k',axis='columns')
target = df['salary_more_then_100k']
print(inputs)
print(target)

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
inputs['company_n']= le_company.fit_transform(inputs['company'])
inputs['job_n']= le_company.fit_transform(inputs['job'])
inputs['degree_n']= le_company.fit_transform(inputs['degree'])
print(inputs.head())