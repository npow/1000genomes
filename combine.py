#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy
import sys
from nolearn.dbn import DBN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import *
from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.externals import joblib
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn import metrics
from utils import *
np.random.seed(42)

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.fn(X)

def load(L):
  return np.concatenate([joblib.load(x) for x in L], axis=1)

def main():
    chromosomes = [5, 18, 19, 20, 21, 22]
    print "Chromosomes: ", chromosomes
    sys.stdout.flush()

    X_train = load(["blobs/X_train_meta_%d.pkl" % c for c in chromosomes])
    X_test = load(["blobs/X_test_meta_%d.pkl" % c for c in chromosomes])
    Y_train = joblib.load('blobs/Y_train.pkl')
    Y_test = joblib.load('blobs/Y_test.pkl')
    print X_train.shape
    print X_test.shape
    print Y_train.shape
    print Y_test.shape
    sys.stdout.flush()

    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_test = le.transform(Y_test)

    clf = Pipeline([
        ('t', StandardScaler()),
#        ('c', DBN([X_train.shape[1], X_train.shape[1], np.unique(Y_train).shape[0]], epochs=10, verbose=1))
        ('c', BaggingClassifier(base_estimator=LinearSVC(class_weight='auto'), n_estimators=100, bootstrap_features=True, max_samples=0.5, max_features=0.5, n_jobs=-1))
    ])
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    
    score = metrics.f1_score(Y_test, pred)
    print("f1-score:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(Y_test, pred))

if __name__ == '__main__':
    main()
