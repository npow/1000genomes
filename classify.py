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
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn import metrics
from utils import *
np.random.seed(42)

PREFIX = "linearsvc_chr%s" % sys.argv[1]
print PREFIX

def load(L):
  return scipy.sparse.hstack([load_sparse_matrix(x) for x in L])

chromosomes = [int(x) for x in sys.argv[1].split(',')]
print chromosomes
sys.stdout.flush()
if len(chromosomes) == 0:
  print "No chromosomes specified"
  sys.exit(1)

X = load(["X_%d" % c for c in chromosomes])
Y = joblib.load('blobs/Y_pop.pkl')
print X.shape
print Y.shape
sys.stdout.flush()

le = LabelEncoder()
Y = le.fit_transform(Y)
joblib.dump(le, 'blobs/le.pkl')

X_base, X_valid, Y_base, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=42)

X_train_meta = []
Y_train_meta = []

skf = StratifiedKFold(Y_base, n_folds=10, shuffle=True, random_state=42)
for train_indices, test_indices in skf:
  X_train, Y_train = X_base[train_indices], Y_base[train_indices]
  X_test, Y_test = X_base[test_indices], Y_base[test_indices]

  clf = LinearSVC()
  clf.fit(X_train, Y_train)
  print "DONE FIT"
  sys.stdout.flush()
  X_train_meta.append(clf.decision_function(X_test))
  Y_train_meta.append(Y_test)

clf = LinearSVC()
clf.fit(X_base, Y_base)
X_test_meta = clf.decision_function(X_valid)

X_train_meta = np.concatenate(X_meta, axis=0)
clf = LinearSVC()
clf.fit(X_train_meta, Y_train_meta)
pred = clf.predict(X_test_meta)

score = metrics.f1_score(Y_valid, pred)
print("f1-score:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(le.inverse_transform(Y_test), le.inverse_transform(pred)))

joblib.dump(X_train_meta, 'blobs/X_train_meta_%s.pkl' % PREFIX)
joblib.dump(Y_train_meta, 'blobs/Y_train_meta_%s.pkl' % PREFIX)
joblib.dump(X_test_meta, 'blobs/X_test_meta_%s.pkl' % PREFIX)
joblib.dump(Y_valid, 'blobs/Y_test_meta_%s.pkl' % PREFIX)
