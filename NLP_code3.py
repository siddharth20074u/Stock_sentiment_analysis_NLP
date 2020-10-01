#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:11:52 2020

@author: siddharthsmac
"""


import pandas as pd

df = pd.read_csv('/users/siddharthsmac/downloads/Data.csv', encoding = 'ISO-8859-1')

train = df[df['Date']<'20150101']
test = df[df['Date']>'20141231']

data_train = train.iloc[:, 2:27]

 
data_train.replace('[^a-zA-Z]',  ' ', regex = True, inplace = True)


list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data_train.columns = new_index


for i in new_index:
    data_train[i] = data_train[i].str.lower()
   
    
headlines_train = []
for row in range(0, len(data_train.index)):
    headlines_train.append(' '.join(str(x) for x in data_train.iloc[row, 0:25]))
    


from sklearn.feature_extraction.text import CountVectorizer
countvector = CountVectorizer(ngram_range = (2,2), max_features = 15000)

train_data = countvector.fit_transform(headlines_train)

from sklearn.ensemble import RandomForestClassifier
randomclassifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy')
randomclassifier.fit(train_data, train['Label'])

test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 0:27]))


test_data = countvector.fit_transform(test_transform)

predictions = randomclassifier.predict(test_data)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
cm = confusion_matrix(test['Label'], predictions)
print(cm)

score = accuracy_score(test['Label'], predictions)
print(score)

report = classification_report(test['Label'], predictions)
print(report)
