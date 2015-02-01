# -*- coding: utf-8 -*-
"""
================================
SVM for Classification
================================
Created on Thu Jan 08 21:40:08 2015
@author: Jiachen
"""
import numpy as np
import pylab as pl
from sklearn import svm, metrics


# load or create the data
#X_train = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
#y_train = np.array([-1,-1,-1,1,1,1,1])

# Note: X_train.shape can check the dimension of the feature loaded
X_train = np.loadtxt('data.train', delimiter=" ")
y_train = np.loadtxt('data.train.label', delimiter=" ")
print 'Training data dim:', X_train.shape
print 'Training label dim:', y_train.shape

X_test = np.loadtxt('data.test', delimiter=" ")
y_test = np.loadtxt('data.test.label', delimiter=" ")

# SVM regularization parameter, for hard margin SVM, 
# select a large C (e.g., 1e10) and check the alphas,
# if all alphas < C, there is no misclassification
C = 0.01

# different SVM kernel model and parameters
# clf = svm.SVC(kernel='linear', shrinking=False, C=C)
# clf = svm.SVC(kernel='poly', degree=3, coef0 = 1.0, shrinking=False, C=C)
# clf = svm.SVC(kernel='rbf', gamma=0.7, shrinking=False, C=C)
# clf = svm.LinearSVC(fit_intercept=False, C=C)
# note: for SVC and NuSVC, they use one-vs-one for multi-class classification,
# so that they will create n_class * (n_class - 1) / 2 classifiers
# LinearSVC implemented one-vs-rest strategy
clf = svm.SVC(kernel='poly', degree=2, coef0=1.0, shrinking=False, C=C)
clf.fit(X_train, y_train)
print 'Learned SVM model:', clf


#Attributes
#    ----------
#    support_ : array-like, shape = [n_SV]
#        Index of support vectors.
#    support_vectors_ : array-like, shape = [n_SV, n_features]
#        Support vectors.
#    n_support_ : array-like, dtype=int32, shape = [n_class]
#        number of support vector for each class.
#    dual_coef_ : array, shape = [n_class-1, n_SV]
#        Coefficients of the support vector in the decision function. \
#        For multiclass, coefficient for all 1-vs-1 classifiers. \
#        The layout of the coefficients in the multiclass case is somewhat \
#        non-trivial. See the section about multi-class classification in the \
#        SVM section of the User Guide for details.
#    coef_ : array, shape = [n_class-1, n_features]
#        Weights assigned to the features (coefficients in the primal
#        problem). This is only available in the case of linear kernel.
#        `coef_` is a readonly property derived from `dual_coef_` and
#        `support_vectors_`
#    intercept_ : array, shape = [n_class * (n_class-1) / 2]
#        Constants in decision function. (截距 b)

print clf.dual_coef_

# Evaluation
# performance on the training data set
y_train_predict = clf.predict(X_train)

print 'Training set acc:', metrics.accuracy_score(y_train, y_train_predict)

print("Classification report on training set:\n%s\n" % (metrics.classification_report(y_train, y_train_predict)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_train, y_train_predict))

# performance on the training data set
y_test_predict = clf.predict(X_test)

print 'Test set acc:', metrics.accuracy_score(y_test, y_test_predict)
print("Classification report on test set:\n%s\n" % (metrics.classification_report(y_test, y_test_predict)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_test_predict))

