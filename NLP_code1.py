#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:37:24 2020

@author: siddharthsmac
"""
import nltk

import pandas as pd

df = pd.read_csv('/users/siddharthsmac/downloads/Data.csv', encoding = 'ISO-8859-1')

train = df[df['Date']<'20150101']
test = df[df['Date']>'20141231']

data_train = train.iloc[:, 2:27]
data_test = test.iloc[:, 2:27]

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wordnet = WordNetLemmatizer()

train_headlines = []
for i in range(0, len(data_train.index)):
    train_headlines.append(' '.join(str(x) for x in data_train.iloc[i, 0:25]))
    
corpus_train = []
for i in range(0, len(train_headlines)):
    review = re.sub('[^a-zA-Z]', ' ', train_headlines[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus_train.append(review)
    
test_headlines = []
for i in range(0, len(data_test.index)):
    test_headlines.append(' '.join(str(x) for x in data_test.iloc[i, 0:25]))
    
corpus_test = []
for i in range(0, len(test_headlines)):
    review = re.sub('[^a-zA-Z]', ' ', test_headlines[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus_test.append(review)
    

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(ngram_range = (1,2), max_features = 100000)
train_data = tf.fit_transform(corpus_train)

from sklearn.ensemble import RandomForestClassifier  
randomclassifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy')  
randomclassifier.fit(train_data, train['Label'])  

test_data = tf.fit_transform(corpus_test)

predictions = randomclassifier.predict(test_data)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
cm = confusion_matrix(test['Label'], predictions)
print(cm)

score = accuracy_score(test['Label'], predictions)
print(score)

report = classification_report(test['Label'], predictions)
print(report)
  
    
