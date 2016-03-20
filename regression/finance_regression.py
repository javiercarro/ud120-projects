#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    

def writeToFile(filename, text, mode):
  with open(filename, mode) as text_file:
    text_file.write(text+"\n")


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.

#########################################
# Regression against salary
#########################################
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)
text = "Slope {0}; Intercept {1}.".format(round(reg.coef_, 3), round(reg.intercept_, 3))
print text
writeToFile("Regression_output.txt", text, "w")

score = reg.score(feature_train, target_train)
text = "Score using training sets: {0}.".format(round(score, 3))
print text
writeToFile("Regression_output.txt", text, "a")

score = reg.score(feature_test, target_test)
text = "Score using test sets: {0}.".format(round(score, 3))
print text
writeToFile("Regression_output.txt", text, "a")
#########################################


#########################################
# Regression against long_term_incentive
#########################################
### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "long_term_incentive"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

reg.fit(feature_train, target_train)
score = reg.score(feature_test, target_test)
text = "Score using long_term_incentive: {0}.".format(round(score, 3))
print text
writeToFile("Regression_output.txt", text, "a")
#########################################

#########################################
# Again Regression against salary
#########################################
### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

# Against test data (with outlier)
reg.fit(feature_test, target_test)
text = "(Using test data, which includes outlier, and long_term_incentive)\n  Slope {0}; Intercept {1}.".format(round(reg.coef_, 3), round(reg.intercept_, 3))
print text
writeToFile("Regression_output.txt", text, "a")
#########################################


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
