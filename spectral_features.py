#! /usr/bin/env python3
# -*- encoding:utf-8 -*-
## Author : Florian Tilquin

import numpy as np
from scipy.optimize import linear_sum_assignment

def rearrange_spectrum(VP_old, VP_new):
    J,S = correlation_assignment(VP_old, VP_new)
    VP_new = VP_new[:,J]*S
    return VP_new

def correlation_assignment(U,V):
    if U.shape[1] != V.shape[1]:
        print('U and V must have the same number of eigenvectors or be shaped the same way')
        return
    A = V.T.dot(U)
    J = linear_sum_assignment(-np.abs(A))[1]
    S = np.sum(U*V[:,J],0)>0
    S = 2*S-1.
    return J,S
