import numpy as np
from scipy.sparse import csr_matrix
from numpy.random import rand
from sklearn.preprocessing import normalize
from misc import diag, no_log
from time import time
from pdb import set_trace


def var_low_bound(docs, z, alpha, beta):
    return 0


def TCVB0(docs, alpha, beta, epsilon=0.0001, log=no_log):
    """Estimates variational word-to-topic assignments for each word in
    a corpus.
    :param docs: scipy sparse DxV matrix where `docs`[d, w] is frequency of
    word `w` in document `d`
    :param alpha: K-dimensional `numpy.array` with topic proportions
    smoothing parameters
    :param beta: topic smoothing parameter (can be scalar or VxK matrix)
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

        #normalize z[d]
        z[d] = normalize(z[d], norm='l1', axis=1)

        #update counts
        #set_trace()
        M = diag(docs[d]).dot(z[d]).toarray()
        N += M
        Nd[d] = M.sum(axis=0) + alpha

        log('document %d/%d preinitialized' % (d + 1, D))

    #sum of array and matrix is matrix, so convertion is required
    N = np.asarray(N) + beta

    #Nt[t] is pre-computed unnormalized expectation topic t
    Nt = np.squeeze(np.asarray(N.sum(axis=0)))
    if type(beta) is float:
        Nt += V * beta
    elif type(beta) is np.ndarray:
        Nt += beta.sum(axis=0)
    else:
        raise 'beta must be either scalar (float) number for symmetric prior or a full matrix VxK for custom prior'

    #do variational updates until convergence
    iteration = 1
    while True:
        iteration_time = time()
        avg_diff = 0.0

        #for each document
        for d in xrange(D):
            #for each word in a document
            max_diff = 0.0
            doc_diff = 0.0

            doc_w = docs.data[docs.indptr[d]:docs.indptr[d + 1]]

            i = 0
            old_z_d = z[d].data.copy()
            #for each word in the document d
            #do variational update and estimate difference
            for w in docs.indices[docs.indptr[d]:docs.indptr[d + 1]]:
                #save old q(z_d) distribution
                old_z = z[d].data[z[d].indptr[w]:z[d].indptr[w + 1]] * doc_w[i]
                #we take expectations ignoring current document and current word
                N[w] -= old_z
                Nt[:] -= old_z
                Nd[d] -= old_z
                #update
                new_z = N[w] / Nt * Nd[d]
                #normalization
                new_z /= new_z.sum()
                #write new values back
                z[d].data[z[d].indptr[w]:z[d].indptr[w + 1]] = new_z
                #expectations update
                new_z *= doc_w[i]
                N[w] += new_z
                Nt[:] += new_z
                Nd[d] += new_z 

                i += 1

                #word_diff = variational_update(d, w)
            doc_diff += np.abs(old_z_d - z[d].data)
            avg_diff += doc_diff.sum()
            max_diff = max(max_diff, doc_diff.max())
            if d % 100 == 0:
                log('document %d/%d was updated' % (d + 1, D))

        avg_diff /= docs.nnz * K
        log('iteration %d. avg diff: %f. max diff: %f. time: %f' % (iteration, avg_diff, max_diff, time() - iteration_time))

        if max_diff < epsilon:
            break

        iteration += 1

    return z
