# _*_ encoding:utf-8 _*_
"""Diffusions Maps"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig,eigh
from dmgraph import graph_laplacian

def diffusion_maps(adjacency, n_components=8, alpha=0.5, eigen_solver=None,
                       random_state=None, eigen_tol=0.0,
                       norm_laplacian=True, drop_first=True):

    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    # print(adjacency.diagonal())
    laplacian = graph_laplacian(adjacency.copy(), alpha=alpha)
    if eigen_solver == 'eigsh':
        lambdas, diffusion_map = eigsh(laplacian, k=n_components,
                                       sigma=None, which='LM',
                                       tol=eigen_tol)
    elif eigen_solver == 'eigh':
        lambdas, diffusion_map = eigh(laplacian)
    else:
        lambdas, diffusion_map = eig(laplacian)
    lambdas = np.real(lambdas)
    diffusion_map = np.real(diffusion_map)
    lambdas = lambdas[n_components::-1]
    maps = diffusion_map[:,n_components::-1] * lambdas[None,:]
    maps_norm = (maps**2).sum(0).reshape(1,-1)
    maps_norm[maps_norm==0] = 1.
    maps = maps/maps_norm

    if drop_first:
        return maps[:,1:n_components], lambdas[1:n_components].reshape(1,-1)
    else:
        return maps[:,:n_components], lambdas[:n_components].reshape(1,-1)
