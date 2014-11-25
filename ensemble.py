#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy
import sys
from itertools import chain, combinations
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
USE_VOTING = False

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.fn(X)

def load(L):
  return np.concatenate([joblib.load(x) for x in L], axis=1)

def all_subsets(ss):
  return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

def main():
    chromosomes = [5, 18, 19, 20, 21, 22]
    print "Chromosomes: ", chromosomes
    sys.stdout.flush()

    Y_train = joblib.load('blobs/Y_train.pkl')
    Y_test = joblib.load('blobs/Y_test.pkl')

    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_test = le.transform(Y_test)

    if USE_VOTING:
        votes = np.zeros((Y_test.shape[0], np.unique(Y_train).shape[0]))
    else:
        X_train_meta = []
        X_test_meta = []
    for cs in all_subsets(chromosomes):
        if cs == ():
            continue
        print cs
        X_train = load(["blobs/X_train_meta_%d.pkl" % c for c in cs])
        X_test = load(["blobs/X_test_meta_%d.pkl" % c for c in cs])

        clf = Pipeline([
            ('t', StandardScaler()),
            ('c', LinearSVC())
#            ('c', DBN([X_train.shape[1], X_train.shape[1], np.unique(Y_train).shape[0]], epochs=10, verbose=1))
        ])
        clf.fit(X_train, Y_train)

        if USE_VOTING:
            pred = clf.predict(X_test)
            for i, p in enumerate(pred):
                votes[i][p] += 1
        else:
            X_train_meta.append(clf.decision_function(X_train))
            X_test_meta.append(clf.decision_function(X_test))

    if USE_VOTING:
        pred = np.zeros((Y_test.shape[0], 1))
        for i, v in enumerate(votes):
            pred[i] = np.argmax(v)
    else:
        X_train_meta = np.concatenate(X_train_meta, axis=1)
        X_test_meta = np.concatenate(X_test_meta, axis=1)
        clf = DBN([X_train_meta.shape[1], X_train_meta.shape[1]//4, X_train_meta.shape[1], np.unique(Y_train).shape[0]], epochs=5, verbose=1, learn_rates=0.01)
        clf.fit(X_train_meta, Y_train)
        pred = clf.predict(X_test_meta)

    score = metrics.f1_score(Y_test, pred)
    print("f1-score:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(Y_test, pred))

if __name__ == '__main__':
    main()
