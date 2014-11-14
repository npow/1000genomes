import numpy as np
import tables as tb
from scipy import sparse
from sklearn.externals import joblib
import h5py
import gzip

def store_sparse_matrix(m, name):
    msg = "This code only works for csr matrices"
    assert(m.__class__ == sparse.csr.csr_matrix), msg
    store = 'blobs/%s.h5' % name
    with tb.openFile(store,'a') as f:
        for par in ('data', 'indices', 'indptr', 'shape'):
            full_name = '%s_%s' % (name, par)
            try:
                n = getattr(f.root, full_name)
                n._f_remove()
            except AttributeError:
                pass

            arr = np.array(getattr(m, par))
            atom = tb.Atom.from_dtype(arr.dtype)
            ds = f.createCArray(f.root, full_name, atom, arr.shape)
            ds[:] = arr

def load_sparse_matrix(name):
    store = 'blobs/%s.h5' % name
    with tb.openFile(store) as f:
        pars = []
        for par in ('data', 'indices', 'indptr', 'shape'):
            pars.append(getattr(f.root, '%s_%s' % (name, par)).read())
    m = sparse.csr_matrix(tuple(pars[:3]), shape=pars[3])
    return m

def get_num_features(file_name):
    f = gzip.open(file_name)
    idx = 0
    for line in f:
        if line[0] == '#':
            continue
        idx += 1
    f.close()
    return idx

