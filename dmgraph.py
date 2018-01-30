"""
Graph utilities and algorithms

Graphs are represented with their adjacency matrices, preferably using
sparse matrices.
"""

import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_array


###############################################################################
# Graph laplacian
def graph_laplacian(csgraph, alpha=0.5, normed=False, return_diag=False):
    """ Return the Laplacian matrix of a directed graph.

    For non-symmetric graphs the out-degree is used in the computation.

    Parameters
    ----------
    csgraph : array_like or sparse matrix, 2 dimensions
        compressed-sparse graph, with shape (N, N).
    normed : bool, optional
        If True, then compute normalized Laplacian.
    return_diag : bool, optional
        If True, then return diagonal as well as laplacian.

    Returns
    -------
    lap : ndarray
        The N x N laplacian matrix of graph.
    diag : ndarray
        The length-N diagonal of the laplacian matrix.
        diag is returned only if return_diag is True.

    Notes
    -----
    The Laplacian matrix of a graph is sometimes referred to as the
    "Kirchoff matrix" or the "admittance matrix", and is useful in many
    parts of spectral graph theory.  In particular, the eigen-decomposition
    of the laplacian matrix can give insight into many properties of the graph.

    For non-symmetric directed graphs, the laplacian is computed using the
    out-degree of each node.
    """

    if normed and (np.issubdtype(csgraph.dtype, np.int)
                   or np.issubdtype(csgraph.dtype, np.uint)):
        csgraph = check_array(csgraph, dtype=np.float64, accept_sparse=True)

    if sparse.isspmatrix(csgraph):
        return _laplacian_sparse(csgraph, alpha=alpha, normed=normed,
                                 return_diag=return_diag)
    else:
        return _laplacian_dense(csgraph, alpha=alpha)


def _laplacian_sparse(graph, alpha=0.5, normed=True, return_diag=False):
    # n_nodes = graph.shape[0]
    if not graph.format == 'coo':
        lap = graph.tocoo()
    else:
        lap = graph.copy()
    diag_mask = (lap.row == lap.col)
    lap.data[diag_mask] = 0
    w = np.asarray(lap.sum(axis=1)).flatten()
    if normed:
        # w = np.sqrt(w)
        w = w**alpha
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap.data /= w[lap.row]
        lap.data /= w[lap.col]
        w = np.asarray(lap.sum(axis=1)).flatten()**0.5
        # lap.data /= w[lap.row]
        lap.data /= w[lap.col]**2
        # lap.data[diag_mask] = (1 - w_zeros[lap.row[diag_mask]]).astype(
            # lap.data.dtype)
    # else:
        # lap.data[diag_mask] = w[lap.row[diag_mask]]
    # lap = 0.5*(lap+lap.T)
    if return_diag:
        return lap.tocsr(), w
    return lap.tocsr()


def _laplacian_dense(graph, alpha=0.5):
    lap = np.asarray(graph)
    n_nodes = lap.shape[0]
    # set diagonal to zero
    # w = -lap.sum(axis=0)
    if lap.shape[0] != lap.shape[1]:
        w0 = lap.sum(axis=0)
        w0[w0<1e-16] = 0
        w0[w0==0] = 1.
        lap[w0==0] = 0
        lap /= w0
    else :
        lap.flat[::n_nodes + 1] = 0
        w = lap.sum(axis=0)
        w = w**alpha
        w[w<1e-16] = 0
        lap[w==0] = 0
        w[w==0] = 1.
        lap /= w
        lap /= w[:, np.newaxis]
        w = lap.sum(0)
        w[w<1e-16] = 0
        lap[:,w==0] = 0
        w[w==0] = 1.
        lap /= w
    return lap
