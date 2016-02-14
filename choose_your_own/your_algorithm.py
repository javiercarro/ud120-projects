#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
import sys
from time import time

def writeToFile(filename, text, mode):
  print text
  with open(filename, mode) as text_file:
    text_file.write(text+"\n")



# Algorithms to try
# 1. k nearest neighors - classic, simple, easy to understand
# 2. Ensemble methods): meta classifiers built from (usually) decision trees
#   2.1. Adaboost (boosting)
#   2.2. Random Forest (averaging)

# 1. k nearest neighors - classic, simple, easy to understand
def KNearestNeigh(k):
  from sklearn.neighbors import KNeighborsClassifier
  
  clf = KNeighborsClassifier(n_neighbors = k)

  start_time = time()
  clf.fit(features_train, labels_train)
  elapsed = time()-start_time
  text = "Training time (kNearestNeigh:{0}): {1}s".format(k, round(elapsed, 3))
  writeToFile("ChooseYourOwn_output.txt", text, "a")

  acc = clf.score(features_test, labels_test)
  text = "Accuracy (kNearestNeigh:{0}): {1}".format(k, round(acc, 3))
  writeToFile("ChooseYourOwn_output.txt", text, "a")

  try:
    prettyPicture(clf, features_test, labels_test)
  except NameError:
    pass

#   2.1. Adaboost (boosting)
def AdaBoost(k):
  from sklearn.ensemble import AdaBoostClassifier

  clf = AdaBoostClassifier(n_estimators = k)

  start_time = time()
  clf.fit(features_train, labels_train)
  elapsed = time()-start_time
  text = "Training time (AdaBoost:{0}): {1}s".format(k, round(elapsed, 3))
  writeToFile("ChooseYourOwn_output.txt", text, "a")

  acc = clf.score(features_test, labels_test)
  text = "Accuracy (AdaBoost:{0}): {1}".format(k, round(acc, 3))
  writeToFile("ChooseYourOwn_output.txt", text, "a")

  #try:
    #prettyPicture(clf, features_test, labels_test)
  #except NameError:
    #pass

#   2.2. Random Forest (averaging)
def RandomForest(k):
  from sklearn.ensemble import RandomForestClassifier

  clf = RandomForestClassifier(n_estimators = k)

  start_time = time()
  clf.fit(features_train, labels_train)
  elapsed = time()-start_time
  text = "Training time (RandomForest:{0}): {1}s".format(k, round(elapsed, 3))
  writeToFile("ChooseYourOwn_output.txt", text, "a")

  acc = clf.score(features_test, labels_test)
  text = "Accuracy (RandomForest:{0}): {1}".format(k, round(acc, 3))
  writeToFile("ChooseYourOwn_output.txt", text, "a")

  #try:
    #prettyPicture(clf, features_test, labels_test)
  #except NameError:
    #pass

text = "File initialization"
writeToFile("ChooseYourOwn_output.txt", text, "w")

#KNearestNeigh(3)
#KNearestNeigh(5)
#KNearestNeigh(10)
#KNearestNeigh(50)

#AdaBoost(10)
#AdaBoost(50)
#AdaBoost(100)
#AdaBoost(200)

RandomForest(10)
#RandomForest(50)
#RandomForest(100)
#RandomForest(200)

