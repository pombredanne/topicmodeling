from scipy.sparse import csr_matrix


def diag(x):
    N = x.shape[1]
    idx = x.nonzero()[1]
    return csr_matrix((x.data, (idx, idx)), shape=(N, N))


def no_log(str):
    print str
