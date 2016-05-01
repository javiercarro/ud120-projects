#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)

def writeToFile(filename, text, mode):
  print text
  with open(filename, mode) as text_file:
    text_file.write(text+"\n")


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
#from sklearn import tree
#clf = tree.DecisionTreeClassifier(min_samples_split=40)
#clf.fit(features_train, labels_train)
#acc = clf.score(features_test, labels_test)
#text = "Accuracy overfit (DecTree): {0}".format(acc)
#writeToFile("FeatureSelection_output.txt", text, "w")


#dt = tree.DecisionTreeClassifier(min_samples_split=40)
#dt.fit(features_train, labels_train)
#print(dt.score(features_train, labels_train))
#print(dt.score(features_test, labels_test))
#print('--------')
#
#for idx, coef in enumerate(dt.feature_importances_):
    #if coef >= 0.2:
        #print(coef)
        #print(idx)
        #print(vectorizer.get_feature_names()[idx])

from sklearn import tree
from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test,pred)
print("Accuracy:",acc)
#for x in range(len(clf.feature_importances_)):
	#if clf.feature_importances_[x] > .2:
		#print x, clf.feature_importances_[x], vectorizer.get_feature_names()[x]
