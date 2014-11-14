import numpy as np
import scipy
import sys
from nolearn.dbn import DBN
from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.externals import joblib
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn import metrics
from utils import *

def load(L):
  return scipy.sparse.hstack([load_sparse_matrix(x) for x in L])

chromosomes = [5, 22]
X = load(["X_%d" % c for c in chromosomes])
Y = joblib.load('blobs/Y_pop.pkl')
print chromosomes
print X.shape
print Y.shape
sys.stdout.flush()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

#clf = LinearSVC(penalty='l2')
clf = RandomForestClassifier(n_estimators=100, n_jobs=4)
if type(clf) in [RandomForestClassifier]:
  clf.fit(X_train.toarray(), Y_train)
  print "DONE FIT"
  sys.stdout.flush()
  pred = clf.predict(X_test.toarray())
else:
  clf.fit(X_train, Y_train)
  print "DONE FIT"
  sys.stdout.flush()
  pred = clf.predict(X_test)

score = metrics.f1_score(Y_test, pred)
print("f1-score:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(Y_test, pred))
