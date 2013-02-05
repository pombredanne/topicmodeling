import numpy as np
from scipy.sparse import csr_matrix
from numpy.random import rand
from sklearn.preprocessing import normalize
import pdb
from misc import diag
from time import time

def no_log(str):
    print str


def TCVB0(docs, alpha, beta, epsilon=0.00001, log=no_log):
    """Estimates variational word-to-topic assignments for each word in
    a corpus.
    :param docs: scipy sparse DxV matrix where `docs`[d, w] is frequency of
    word `w` in document `d`
    :param alpha: K-dimensional `numpy.array` with topic proportions
    smoothing parameters
    :param beta: topic smoothing parameter
    :param epsilon: minimal update value for convergence detection
    """
    D, V = docs.shape
    K = len(alpha)

    #store variational q_{z_{d,w} = t} for each d as sparse table in
    #array z
    z = np.zeros(D, dtype=object)

    #initialize counts
    #N[t, w] = expectaction of unnormalized phi_{k,w}
    N = np.zeros((V, K), dtype=float)

    #Nd[d, t] = unnormalized theta_{d,k}
    Nd = np.zeros((D, K), dtype=float)

    for d in xrange(D):
        #random initialization
        init = rand(docs[d].nnz * K)
        active_words = docs[d].nonzero()[1]
        ij = (np.repeat(active_words, K), np.tile(np.arange(K), len(active_words)))

        #z[d] is VxK sparse row matrix
        z[d] = csr_matrix((init, ij), shape=(V, K))
        #N += z[d]

        #normalize z[d]
        z[d] = normalize(z[d], norm='l1', axis=1)

        #update counts
        N += diag(docs[d]).dot(z[d]).toarray()
        Nd[d] = z[d].sum(axis=0) + alpha

        log('document %d/%d preinitialized' % (d + 1, D))

    #sum of array and matrix is matrix, so convertion is required
    N = np.asarray(N) + beta

    #Nt[t] is pre-computed unnormalized expectation topic t
    Nt = np.squeeze(np.asarray(N.sum(axis=0))) + V * beta

    #do variational updates until convergence
    iteration = 1
    while True:
        """Performs variational update for document `d` and word `w`
        """
        def variational_update(d, w):
            old_z = z[d][w].data
            #we take expectations ignoring current document and current word
            N[w, :] -= old_z
            Nt[:] -= old_z
            Nd[d] -= old_z
            #update
            new_z = old_z.copy()
            new_z = N[w] / Nt * Nd[d]
            #normalization
            new_z /= new_z.sum()
            #write new values back
            z[d].data[z[d].indptr[w]:z[d].indptr[w + 1]] = new_z
            #counts update
            #new_z = z[d][w].data
            N[w, :] += new_z
            Nt[:] += new_z
            Nd[d] += new_z

            return np.max(np.abs(old_z - new_z))

        iteration_time = time()
        #for each document
        for d in xrange(D):
            #for each word in a document
            max_diff = 0.0
            doc_time = time()
            for w in docs[d].nonzero()[1]:
                #do variational update and estimate max difference
                max_diff = np.max(variational_update(d, w), max_diff)
            log('document %d/%d was updated. max diff is %f. time: %f' % (d + 1, D, max_diff, time() - doc_time))

        log('iteration %d. max diff is %f. time: %f' % (iteration, max_diff, time() - iteration_time))

        if max_diff < epsilon:
            break

        iteration += 1

    return z
