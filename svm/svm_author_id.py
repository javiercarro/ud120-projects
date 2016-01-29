#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

def writeToFile(filename, text, mode):
  with open(filename, mode) as text_file:
    text_file.write(text+"\n")


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




from sklearn import svm


#########################################################
### SVC Linear - All samples ###

def SVCLinearAllSamples():

  clf = svm.SVC(kernel='linear')

  start_time = time()
  clf.fit(features_train, labels_train)
  elapsed = time()-start_time
  text = "Training time (SVC linear all samples): {0}s".format(round(elapsed, 3))
  print text
  writeToFile("SVC_output.txt", text, "w")

  start_time_train = time()
  predict = clf.predict(features_test)
  elapsed = time()-start_time
  text = "Predict time (SVC linear all samples): {0}s".format(round(elapsed, 3))
  print text
  writeToFile("SVC_output.txt", text, "a")

  acc = clf.score(features_test, labels_test)
  text = "Accuracy (SVC linear all samples): {0}%".format(acc * 100.)
  print text
  writeToFile("SVC_output.txt", text, "a")

#########################################################


#########################################################
### SVC Linear - Using 1% of samples ###

def SVCLinearOnePercentSamples():

  features_train_1 = features_train[:len(features_train)/100] 
  labels_train_1 = labels_train[:len(labels_train)/100] 

  clf = svm.SVC(kernel='linear')

  start_time = time()
  clf.fit(features_train_1, labels_train_1)
  elapsed = time()-start_time
  text = "Training time (SVC linear 1% samples): {0}s".format(round(elapsed, 3))
  print text
  writeToFile("SVC_output.txt", text, "a")

  start_time_train = time()
  predict = clf.predict(features_test)
  elapsed = time()-start_time
  text = "Predict time (SVC linear 1% samples): {0}s".format(round(elapsed, 3))
  print text
  writeToFile("SVC_output.txt", text, "a")

  acc = clf.score(features_test, labels_test)
  text = "Accuracy (SVC linear 1% samples): {0}%".format(acc * 100.)
  print text
  writeToFile("SVC_output.txt", text, "a")

#########################################################


#########################################################
### SVC RBF - Using 1% of samples ###

def SVC_RBF_OnePercentSamples():

  features_train_1 = features_train[:len(features_train)/100] 
  labels_train_1 = labels_train[:len(labels_train)/100] 

  # 1 is the default value for C
  C = [1, 10, 100, 1000, 10000]

  for c_param in C:

    clf = svm.SVC(kernel='rbf', C=c_param)

    start_time = time()
    clf.fit(features_train_1, labels_train_1)
    elapsed = time()-start_time
    text = "Training time (SVC RBF 1% samples; C={}): {}s".format(c_param, round(elapsed, 3))
    print text
    writeToFile("SVC_output.txt", text, "a")

    start_time_train = time()
    predict = clf.predict(features_test)
    elapsed = time()-start_time
    text = "Predict time (SVC RBF 1% samples): {0}s".format(round(elapsed, 3))
    print text
    writeToFile("SVC_output.txt", text, "a")

    acc = clf.score(features_test, labels_test)
    text = "Accuracy (SVC RBF 1% samples): {0}%".format(acc * 100.)
    print text
    writeToFile("SVC_output.txt", text, "a")

#########################################################

#########################################################
### SVC RBF - OptimizedC All Samples ###

def SVC_RBF_OptimizedC_AllSamples():

  # 1 is the default value for C
  c_param = 10000

  clf = svm.SVC(kernel='rbf', C=c_param)

  start_time = time()
  clf.fit(features_train, labels_train)
  elapsed = time()-start_time
  text = "Training time (SVC RBF All samples; C={}): {}s".format(c_param, round(elapsed, 3))
  print text
  writeToFile("SVC_output.txt", text, "a")

  start_time_train = time()
  predict = clf.predict(features_test)
  elapsed = time()-start_time
  text = "Predict time (SVC RBF All samples): {0}s".format(round(elapsed, 3))
  print text
  writeToFile("SVC_output.txt", text, "a")

  acc = clf.score(features_test, labels_test)
  text = "Accuracy (SVC RBF All samples): {0}%".format(acc * 100.)
  print text
  writeToFile("SVC_output.txt", text, "a")

#########################################################

##########################################################
### SVC RBF - C=10000, 1% Samples, elements 10, 26, 50 ###

def SVC_RBF_OptimizedC_OnePercentSamples_Elements():

  # 1 is the default value for C
  c_param = 10000

  clf = svm.SVC(kernel='rbf', C=c_param)

  # Use _1 sufix to work with 1% of samples
  features_train_1 = features_train[:len(features_train)/100] 
  labels_train_1 = labels_train[:len(labels_train)/100] 

  start_time = time()
  # Use _1 sufix to work with 1% of samples
  clf.fit(features_train, labels_train)
  elapsed = time()-start_time
  text = "Training time (SVC RBF 1% samples; C=10000): {}s".format(round(elapsed, 3))
  print text
  #writeToFile("SVC_output.txt", text, "a")

  start_time_train = time()
  predict = clf.predict(features_test)
  elapsed = time()-start_time
  text = "Predict time (SVC RBF 1% samples): {0}s".format(round(elapsed, 3))
  print text
  #writeToFile("SVC_output.txt", text, "a")

  #text = "Prediction for element 10: {0}".format(predict[10])
  #print text
  #writeToFile("SVC_output.txt", text, "a")
  #text = "Prediction for element 26: {0}".format(predict[26])
  #print text
  #writeToFile("SVC_output.txt", text, "a")
  #text = "Prediction for element 50: {0}".format(predict[50])
  #print text
  #writeToFile("SVC_output.txt", text, "a")
  text = "Chris cases: {0}".format(sum(predict))
  print text
  writeToFile("SVC_output.txt", text, "a")

#########################################################

#SVCLinearAllSamples()

#SVCLinearOnePercentSamples()

#SVC_RBF_OnePercentSamples()

#SVC_RBF_OptimizedC_AllSamples()

#SVC_RBF_OptimizedC_OnePercentSamples_Elements()

