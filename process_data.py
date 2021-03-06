import numpy as np
from scipy import sparse
from sklearn.externals import joblib
from utils import *
import gzip
import sys

DUMP_LABELS = False
DUMP_VCF = True
if DUMP_VCF:
    CHR_NUM = sys.argv[1]
    print CHR_NUM
    FILE_NAME = 'data/ALL.chr%s.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.vcf.gz' % CHR_NUM
    NUM_SAMPLES = 2504
    NUM_F = get_num_features(FILE_NAME)
    print "NUM_F: %d" % NUM_F

    def nnz(s):
        A = s[0]
        B = s[2]
        if A == '0' and B == '0':
            return 0
        elif A == '0' and B != '0':
            return 1
        elif A != '0' and B == '0':
            return 1
        else:
            return 2

    idx = 0
    X = sparse.lil_matrix((NUM_F, NUM_SAMPLES))
    f = gzip.open(FILE_NAME)
    for line in f:
        if line[0] == '#':
            continue
        if idx % 10000 == 0:
            print idx
        line = line.split('\t')[9:]
        for j, x in enumerate(line):
            v = nnz(x)
            if v > 0:
                X[idx, j] = v
        idx += 1
    print X.shape
    f.close()
    store_sparse_matrix(X.T.tocsr(), 'X_%s' % CHR_NUM)
    sys.exit(0)

if DUMP_LABELS:
    f = open('labels.txt')
    Y_pop = []
    Y_superpop = []
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.split('\t')
        Y_pop.append(line[1].strip())
        Y_superpop.append(line[2].strip())
    Y_pop = np.array(Y_pop).T
    Y_superpop = np.array(Y_superpop).T
    print Y_pop.shape
    print Y_superpop.shape
    joblib.dump(Y_pop, 'blobs/Y_pop.pkl')
    joblib.dump(Y_superpop, 'blobs/Y_superpop.pkl')
    sys.exit(0)
