import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
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
    #array z
    z = np.zeros(D, dtype=object)

    #initialize counts
    #N[t, w] = expectaction of phi_{k,w}
    N = np.zeros((K, V), dtype=float)

    #Nd[d, t] = unnormalized theta_{d,k}
    Nd = np.zeros((D, K), dtype=float)

    for d in xrange(D):
        #random initialization
        init = randn(docs[d].nnz * K)
        ij = (None, None)
        ij[0] = np.tile(docs[d].nonzero()[1], K)
        ij[1] = np.tile(np.arange(K), V)

        #z[d] is VxK sparse row matrix
        z[d] = csc_matrix((init, ij), shape=(V, K), dtype=float)

        #normalize z[d]
        z[d] = normalize(z[d], norm='L1', axis=1)

        for t in xrange(K):
            N[t] += docs[d].multiply(z[d][:, t])
            Nd[t] = np.dot(docs[d], z[d][:, t])

        z[d] = z[d].to_csr()

    #Nt[t] is pre-computed expectation topic t
    Nt = N.sum(axis=1)

    #do variational updates until convergence
    while True:
        """Performs variational update for document `d` and word `w`
        """
        def variational_update(d, w):
            old_z = z[d][w]
            #we take expectations ignoring current document and current word
            N[:, w] -= old_z
            Nt -= old_z
            Nd[d] -= old_z
            #update
            z[d][w] = (N[:, w] + beta) / (Nt + V * beta) \
                * (Nd[d] + alpha)
            #normalization
            z[d][w] /= z[d][w].sum()
            #counts update
            N[:, w] += z[d][w]
            Nt += z[d, w]
            Nd[d] += z[d, w]

            return np.max(np.abs(old_z - z[d][w]))

        max_diff = 0.0
        for d in xrange(D):
            for w in docs[d].nonzero()[0]:
                max_diff = np.max(variational_update(d, w), max_diff)

        if max_diff < epsilon:
            break

    #make theta from Nd and phi from N
    phi = N / N.sum(axis=1)[:, np.newaxis]
    theta = Nd / Nd.sum(axis=1)[:, np.newaxis]

    return phi, theta, z
