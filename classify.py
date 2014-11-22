#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from nolearn.dbn import DBN
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

USE_STACKING = True

if len(sys.argv) == 1:
  print "No chromosomes specified"
  sys.exit(1)

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.fn(X)

def load(L):
  return scipy.sparse.hstack([load_sparse_matrix(x) for x in L])

def main():
    chromosomes = [int(x) for x in sys.argv[1].split(',')]
    print "Chromosomes: ", chromosomes
    sys.stdout.flush()

    X = load(["X_%d" % c for c in chromosomes])
    Y = joblib.load('blobs/Y_pop.pkl')
    print X.shape
    print Y.shape
    sys.stdout.flush()

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    joblib.dump(le, 'blobs/le.pkl')

    if USE_STACKING:
        skf = StratifiedKFold(Y, n_folds=4, shuffle=True, random_state=42)
        for train_indices, test_indices in skf:
            X_train, Y_train = X[train_indices], Y[train_indices]
            X_test, Y_test = X[test_indices], Y[test_indices]

            clf = Pipeline([
                ('t', Transformer(LinearSVC().fit(X_train, Y_train).decision_function)),
                ('c', LinearSVC()),
            ])

            clf.fit(X_train, Y_train)
            print "DONE FIT"
            sys.stdout.flush()

            pred = clf.predict(X_test)

            score = metrics.f1_score(Y_test, pred)
            print("f1-score:   %0.3f" % score)

            print("classification report:")
            print(metrics.classification_report(le.inverse_transform(Y_test), le.inverse_transform(pred)))
    else:
        X_base, X_valid, Y_base, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=42)
        clf = LinearSVC()
        clf.fit(X_base, Y_base)
        pred = clf.predict(X_valid)

        score = metrics.f1_score(Y_valid, pred)
        print("f1-score:   %0.3f" % score)

        print("classification report:")
        print(metrics.classification_report(le.inverse_transform(Y_valid), le.inverse_transform(pred)))

if __name__ == '__main__':
    main()
