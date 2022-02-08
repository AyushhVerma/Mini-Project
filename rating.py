# Amazon Data Rating Review Analysis and Predicting rating from review using Linear SVM Algorithm


# data preprocessing
import numpy as np
import pandas as pd

dataset = pd.read_csv(r'D:\archive\reviews.csv')

dataset = dataset[dataset['reviews.text'] != '']
dataset = dataset[dataset['reviews.rating'].notnull()]

df = dataset[dataset['reviews.rating'] == 5]
df = df.sample(frac=1).reset_index(drop=True)
df = df[:len(df)//20]

dataset = pd.concat([dataset[dataset['reviews.rating'] != 5], df])
dataset = dataset.sample(frac=1).reset_index(drop=True)[:3000]
dataset['reviews.text'] = dataset['reviews.text'] + ' ' + dataset['reviews.title']
X = dataset[['reviews.text']]
y = dataset[['reviews.rating']]


# Natural Language Processing
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, len(X)):
    review = str(re.sub('[^a-zA-Z]', ' ', str(X["reviews.text"][i])))
    review = review.lower().split()
    
    ps = PorterStemmer()
    
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    
    corpus.append(review)

# feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# cv = CountVectorizer()
tfidf = TfidfVectorizer()
# X = cv.fit_transform(corpus).toarray()
X_tfidf = tfidf.fit_transform(corpus).toarray()
print(X_tfidf.shape)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size = 0.33)

# column vector to 1-d array
y_train = np.array(y_train['reviews.rating'])
y_test = np.array(y_test['reviews.rating'])

# linear svr model
from sklearn.svm import SVR
model = SVR(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# shows prediction and true values side by side
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# accuracy function
def accuracy_score(actual, pred):
    return (1 - np.sum(abs(pred - actual)) / np.sum(actual)) * 100

# accuracy of model
print(accuracy_score(y_test, y_pred))


