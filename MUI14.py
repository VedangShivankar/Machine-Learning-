import pandas as pd
df = pd.read_csv("spam14.csv")
print(df.head())
df.groupby('Catrgory').describe()
df['spam']= df['Category'].apply(lambda x:1 if x=='spam' else 0)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(inputs,test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_test.values)
X_train_count.toarray()[:3]

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count, y_train)

emails = [
'Hey jack, can we go to watch baseball tomorrow?'
'Get 50% off on black friday deals, dont miss it'
]
emails.count = v.transform(emails)
model.predict(emails_count)

X_test_count = v.transform(X_test)
model.score(X_test_count,y y_test)