#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()





#########################################################
### your code goes here ###

def writeToFile(filename, text, mode):
  print text
  with open(filename, mode) as text_file:
    text_file.write(text+"\n")


from sklearn import tree

def accDecTree():
  clf = tree.DecisionTreeClassifier(min_samples_split = 40)

  start_time = time()
  clf.fit(features_train, labels_train)
  elapsed = time()-start_time
  text = "Training time (DecTree): {0}s".format(round(elapsed, 3))
  writeToFile("DecTree_output.txt", text, "w")

  acc = clf.score(features_test, labels_test)
  text = "Accuracy (DecTree): {0}s".format(round(acc, 3))
  writeToFile("DecTree_output.txt", text, "a")

def featuresOriginal():
  n_features = len(features_train[0])
  text = "Features (Original): {0}".format(round(n_features, 3))
  writeToFile("DecTree_output.txt", text, "a")

def accDecTree_1PercentFeatures():
  clf = tree.DecisionTreeClassifier(min_samples_split = 40)

  start_time = time()
  clf.fit(features_train, labels_train)
  elapsed = time()-start_time
  text = "Training time (DecTree 1%features): {0}s".format(round(elapsed, 3))
  writeToFile("DecTree_output.txt", text, "a")

  acc = clf.score(features_test, labels_test)
  text = "Accuracy (DecTree 1%features): {0}s".format(round(acc, 3))
  writeToFile("DecTree_output.txt", text, "a")

#########################################################

#accDecTree()
#featuresOriginal()
accDecTree_1PercentFeatures()

#########################################################
