import time
from itertools import izip

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

def coo_submatrix_pull(matr, rows, cols):
    """
    Pulls out an arbitrary i.e. non-contiguous submatrix out of
    a sparse.coo_matrix. 
    
    This should be incorporated into SciPy at some point, methinks.
    """
    if type(matr) != sparse.coo_matrix:
        raise TypeError('Matrix must be sparse COOrdinate format')
    
    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1])
    
    lr = len(rows)
    lc = len(cols)
    
    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return sparse.coo_matrix((matr.data[newelem], np.array([gr[newrows], \
        gc[newcols]])), (lr, lc))


def make_ktk_ktt(labels, kernels, usebias=False):
    """
    Calculation of the two matrices needed to do the linear regression:
    KtK:  the gram matrix containing the dot products of the kernels
    KtT:  the dot products of the kernels with the target values
    
    Returns KtK, KtT and goodidx, the last being an array of indices
    for the non-zero rows and columns of KtK (KtK and KtT exclude
    the zero-only rows and columns, so goodidx is needed to map back
    to original kernel indices).
    """
    ix_pos, = (labels == 1).nonzero()
    ix_neg, = (labels == -1).nonzero()
    ub_int = int(usebias)
    n_pos = len(ix_pos)
    n_neg = len(ix_neg)
    numgenes = len(labels)
    numkernels = len(kernels)
    eps = np.finfo(np.float64).eps
    pos_const = 2.*n_neg / (n_pos + n_neg)
    neg_const = -2.*n_pos / (n_pos + n_neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError('Must have at least one positive and one negative')
    n_unlabeled = (labels == 0).sum()
    if n_pos + n_neg + n_unlabeled != len(labels):
        raise ValueError('Elements of Y must be either -1, 0, or 1')
    if numgenes == 0:
        raise ValueError('Empty target vector!')
    if numkernels == 0:
        raise ValueError('Empty kernel set!')
    KtK = np.zeros((numkernels + ub_int, numkernels + ub_int))
    KtT = np.zeros(numkernels + ub_int)

    Wpp = [None] * numkernels    # temp storage of +/+ non-diagonal affinities
    Wpn = [None] * numkernels    # temp storage of +/- elements affinities
    nPpElem = n_pos * (n_pos - 1) # # +/+ elements that aren't diagonal
    nPnElem = 2 * n_pos * n_neg   # # +/- elements
    ppTarget = pos_const**2           # target for +/+ elements
    pnTarget = pos_const * neg_const;  # target for +/- elements
    elem = (nPpElem + nPnElem)
    # value of the bias interactions
    if usebias:
        biasVal = 1. / (nPpElem + nPnElem)
        KtT[0] = biasVal * (ppTarget * nPpElem + pnTarget * nPnElem)
        KtK[0, 0] = biasVal
    
    for ii in xrange(numkernels):
        # Pull out the positive-positive and positive-negative matrices
        # and convert them to compressed sparse column (CSC) for easy
        # elementwise multiplication with one another.
        Wpp[ii] = coo_submatrix_pull(kernels[ii], ix_pos, ix_pos)
        
        # Zero out the diagonal
        Wpp[ii].data[Wpp[ii].row == Wpp[ii].col] = 0.

        # Convert
        Wpp[ii] = Wpp[ii].tocsc()
        
        Wpn[ii] = coo_submatrix_pull(kernels[ii], ix_pos, ix_neg)
        Wpn[ii] = Wpn[ii].tocsc()

        ssWpp = Wpp[ii].sum()
        ssWpn = Wpn[ii].sum()

        KtT[ii + ub_int] = ppTarget * ssWpp + 2. * pnTarget * ssWpn;
        if usebias:
            KtK[ii + ub_int, 0] = biasVal * (ssWpp + 2. * ssWpn)
            KtK[0, ii + ub_int] = KtK[ii + ub_int, 0]

        for jj in xrange(ii + 1):
            KtK[ii + ub_int, jj + ub_int] = \
                    (Wpp[ii].multiply(Wpp[jj])).sum() + \
                    2. * (Wpn[ii].multiply(Wpn[jj])).sum()
            
            # Make KtK symmetric.
            KtK[jj + ub_int, ii + ub_int] = KtK[ii + ub_int, jj + ub_int]
 
    ss = np.abs(KtK).sum(axis=0)
    goodidx, = (ss > (eps * max(ss))).nonzero()
    KtK = KtK[np.ix_(goodidx, goodidx)]
    KtT = KtT[goodidx]

    if not usebias:
        # TODO: make this part of the loop. Might be faster.
        means = np.array([(Wpp[i].mean() * nPpElem + Wpn[i].mean() * \
                           nPnElem) / (nPpElem + nPnElem) for i in goodidx])
        KtK -= np.outer(means, means) * (nPpElem + nPnElem)
        KtT -= means * (ppTarget * nPpElem + pnTarget * nPnElem)
    
    return KtK, KtT, goodidx

def find_kernel_weights(labels, kernels, usebias=False):
    KtK, KtT, goodidx = make_ktk_ktt(labels, kernels, usebias)
    ub_int = int(usebias)
    indices = np.arange(len(goodidx) - ub_int)
    done = False
    
    while not done:
        if len(indices) == 0:
            break
        try:
            alpha = np.linalg.solve(KtK, KtT)
        
        except np.linalg.LinAlgError:
            # Couldn't invert KtK, just break out of the while loop.
            indices = []
            break
        if np.any(np.isnan(alpha)):
            indices = []
            break

        # Find the locations of all the negative weights
        negWeights, = (alpha < 0).nonzero()
        
        # A negative bias is okay
        if usebias:
            negWeights = np.setdiff1d(negWeights, [0])

        if len(negWeights) > 0:

            # Save ourselves some computation by just eliminating rows and cols
            # from K^T K and elements from K^T y. The savings from this appear
            # to be nominal, though.
            retain = np.setdiff1d(range(len(alpha)), negWeights)
            KtK = KtK[np.ix_(retain, retain)]
            KtT = KtT[retain]

            # Remove from the indices array any kernels that were removed.
            indices = np.setdiff1d(indices, indices[negWeights - ub_int])
        else:
            done = True
            print "%d kernels chosen, indices %s, weights %s" % \
                    ((len(KtT) - ub_int), str(goodidx[indices]),
                     str(alpha[ub_int:]))
    
    # Undo our previous trickery
    indices = goodidx[indices]

    # if we have no non-zero alpha entries, use the average kernel
    if len(indices) == 0:
        indices = np.arange(len(kernels));
        alpha = np.array([0] * ub_int + [1./len(kernels)] * len(kernels))
        print "All kernels eliminated or empty, using average kernel."

    return indices, alpha[ub_int:]

def score(labels, kernel, labelbias=True):
    """
    Given a (combined) kernel and a label vector, do label propagation,
    optionally with GeneMANIA's label biasing scheme for small positive
    sets.
    """
    tstart = time.clock()
    if labelbias:
        numpos = (labels == 1).sum()
        numneg = (labels == -1).sum()
        labels[labels == 0] = (numpos - numneg) / (numpos + numneg)

    kernel = kernel.tocoo(copy=True)

    colsums = np.asarray(kernel.sum(axis=0)).squeeze()
    diag = 1. / np.sqrt(colsums + np.finfo(np.float64).eps)
    kernel.data *= diag[kernel.row] * diag[kernel.col]
    
    numelem = len(diag)
    diag_indices = np.concatenate((np.arange(numelem).reshape(1, numelem),
                                   np.arange(numelem).reshape(1, numelem)),
                                  axis=0)
    normalizer_elems = 1 + np.asarray(kernel.sum(axis=0)).squeeze()
    normalizer = sparse.coo_matrix((normalizer_elems, diag_indices))
    laplacian = normalizer - kernel
    laplacian = (laplacian + laplacian.T) / 2.

    discriminant, info = splinalg.cg(laplacian, labels)
    return discriminant

def predict(labels, kernels, weighting=None):
    """
    Weight the networks (unless weighting='equal', then just use an 
    equal blending of all) and run label propagation on the resulting
    combined network.
    """
    if weighting == 'equal':
        selected = np.arange(len(kernels))
        weights = np.ones(len(kernels), dtype=float) / len(kernels)
    else:
        selected, weights = find_kernel_weights(labels, kernels)

    combined = sparse.csc_matrix(kernels[0].shape, dtype=np.float64)
    
    for idx in xrange(len(selected)):
        combined = combined + weights[idx] * kernels[selected[idx]].tocsc()

    discriminant = score(labels, combined)
    return discriminant, selected, weights
