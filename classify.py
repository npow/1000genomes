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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
joblib.dump(Y_test, 'blobs/Y_test.pkl')

select = SelectKBest(chi2, k=1000)
X_train = select.fit_transform(X_train, Y_train)
X_test = select.transform(X_test)
print X_train.shape
print X_test.shape

clf = LinearSVC(penalty='l2')
#clf = RandomForestClassifier(n_estimators=3000, n_jobs=4)
if type(clf) in [RandomForestClassifier]:
  clf.fit(X_train.toarray(), Y_train)
  print "DONE FIT"
  sys.stdout.flush()
  pred = clf.predict(X_test.toarray())
  dist = clf.decision_function(X_test.toarray())
else:
  clf.fit(X_train, Y_train)
  print "DONE FIT"
  sys.stdout.flush()
  pred = clf.predict(X_test)
  dist = clf.decision_function(X_test)

joblib.dump(pred, 'blobs/%s_pred.pkl' % PREFIX)
joblib.dump(dist, 'blobs/%s_dist.pkl' % PREFIX)

score = metrics.f1_score(Y_test, pred)
print("f1-score:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(le.inverse_transform(Y_test), le.inverse_transform(pred)))
