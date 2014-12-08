#!/usr/bin/env python

from __future__ import division
import matplotlib.pyplot as plt
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
    target = ''
    chromosomes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 22]
    print "Chromosomes: ", chromosomes
    sys.stdout.flush()

    X_train = load(["blobs/X_train_meta_rand_%d%s.pkl" % (c, target) for c in chromosomes])
    X_test = load(["blobs/X_test_meta_rand_%d%s.pkl" % (c, target) for c in chromosomes])
    Y_train = joblib.load('blobs/Y_train%s.pkl' % target)
    Y_test = joblib.load('blobs/Y_test%s.pkl' % target)
    print X_train.shape
    print X_test.shape
    print Y_train.shape
    print Y_test.shape
    sys.stdout.flush()

    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_test = le.transform(Y_test)

    estimator = BaggingClassifier(base_estimator=LinearSVC(), n_estimators=100, bootstrap_features=False, max_samples=1.0, max_features=1.0)
    clf = Pipeline([
        ('t', StandardScaler()),
#        ('c', DBN([X_train.shape[1], X_train.shape[1], np.unique(Y_train).shape[0]], epochs=10, verbose=1))
        ('c', estimator)
    ])
    print estimator.get_params()
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    
    score = metrics.f1_score(Y_test, pred)
    print("f1-score:   %0.3f" % score)

    le = joblib.load('blobs/le%s.pkl' % target)
    print("classification report:")
    print(metrics.classification_report(le.inverse_transform(Y_test), le.inverse_transform(pred)))
    cm = metrics.confusion_matrix(le.inverse_transform(Y_test), le.inverse_transform(pred))
    #print(cm)

    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cm.jpg')

if __name__ == '__main__':
    main()
