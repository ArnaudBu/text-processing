# !/usr/bin/env python
# -*- coding: utf-8 -*-

from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn.sparse_dot_topn as ct
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from utils import cln


# N-grams for vectorizer
def ngrms(s, n=[2,3], sp_val=' '):
    a = []
    for i in n:
        b = zip(*[s[j:] for j in range(i)])
        a += [''.join(c) for c in b if sp_val not in c] * i
    return a


# Matrix multiplication with top values
def awsm_cossim_top(A, B, ntop=10, lower_bound=0):
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
    idx_dtype = np.int32
    nnz_max = M*ntop
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)
    return csr_matrix((data, indices, indptr), shape=(M, N))


# Vectorizer
def vctrz(n, fn_cln=cln, fn_anl=ngrms):
    n_clean = [fn_cln(a) for a in n]
    v = TfidfVectorizer(analyzer=fn_anl)
    v.fit(n_clean)
    mat = v.transform(n_clean)
    return n_clean, v, mat


# Matching
def match(n, v, mat, ns, t=10, fn_cln=cln):
    n_cl = [fn_cln(a) for a in n]
    d = v.transform(n_cl)
    m = awsm_cossim_top(d, mat.transpose(), t)
    nz = m.nonzero()
    sparserows = nz[1]
    sparsecols = nz[0]
    nr_matches = sparsecols.size
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    sim_cosine = np.zeros(nr_matches)
    for i in range(0, nr_matches):
        left_side[i] = ns[sparserows[i]]
        right_side[i] = n[sparsecols[i]]
        sim_cosine[i] = m.data[i]
    return pd.DataFrame({'entry': right_side,
                         'match': left_side,
                         'sim_cosine': sim_cosine})
