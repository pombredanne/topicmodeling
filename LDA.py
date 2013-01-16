import numpy as np
from scipy.sparse import csr_matrix
from numpy.random import randn
from sklearn.preprocessing import normalize


def TCVB0(docs, alpha, beta, epsilon=0.00001):
    """Estimates variational word-to-topic assignments for each word in
    a corpus.
    :param docs: scipy sparse DxV matrix where `docs`[d, w] is frequency of
    word `w` in document `d`
    """
    D, V = docs.shape
    K = len(alpha)

    #store variational q_{z_{d,w} = t} for each d as sparse table in
    #array q_z
    q_z = np.zeros(D, dtype=object)

    #initialize counts
    N = np.zeros((K, V), dtype=float)

    for d in xrange(D):
        #random initialization
        init = randn(docs[d].nnz * K)
        ij = (None, None)
        ij[0] = np.tile(docs[d].nonzero()[1], K)
        ij[1] = np.tile(np.arange(K), V)

        #q_z[d] is VxK sparse row matrix
        q_z[d] = csr_matrix((init, ij), shape=(V, K), dtype=float)

        #normalize q_z[d]
        q_z[d] = normalize(q_z[d], norm='L1', axis=1)

        for t in xrange(K):
            N[t] += docs[d].transpose(copy=False).multiply(q_z[d][:, t].transpose(copy=False))       

    #do variational updates until convergence
    while True:
        def variational_update(d, w):
            
