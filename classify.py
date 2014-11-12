import numpy as np
import scipy
import sys
from nolearn.dbn import DBN
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA, RandomizedPCA
from sklearn.externals import joblib
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn import metrics

X_21 = joblib.load('blobs/X_21.pkl')
X_22 = joblib.load('blobs/X_22.pkl')
X = scipy.sparse.hstack((X_21, X_22))
del X_21
del X_22
Y = joblib.load('blobs/Y_pop.pkl')
print X.shape
print Y.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

clf = LinearSVC()
clf.fit(X_train, Y_train)
print "DONE FIT"
pred = clf.predict(X_test)

score = metrics.f1_score(Y_test, pred)
print("f1-score:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(Y_test, pred))
