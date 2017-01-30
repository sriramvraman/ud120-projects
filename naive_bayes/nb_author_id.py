#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf = GaussianNB()

start_time_0 = time()
clf.fit(features_train, labels_train)
time_taken_to_fit = time() - start_time_0

start_time_1 = time()
pred = clf.predict(features_test)
time_taken_to_predict = time() - start_time_1

acc = accuracy_score(labels_test, pred)

print "Accuracy = " + str(round(acc,4))
print "Time taken to fit = " + str(round(time_taken_to_fit, 3)) + " s"
print "Time taken to predict = " + str(round(time_taken_to_predict, 3)) + " s"


#########################################################


