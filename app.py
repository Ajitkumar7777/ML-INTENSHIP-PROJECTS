import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

cv = CountVectorizer()
X = cv.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

msg = ["Win money now!!!"]
msg_vec = cv.transform(msg)
print("Prediction:", model.predict(msg_vec)[0])
