#! /usr/bin/env python3
# -*- encoding:utf-8 -*-

import numpy as np
import numpy.random as rd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
# from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

Score = np.load('Score.npy')
MF = np.load('Match_features.npy')
MF[np.isnan(MF)] = np.nanmean(MF)
L = []
mf = MF.copy()
# for n_c in range(1,80):
# pca = PCA(n_components=n_c)
#    pca = Isomap(n_components=n_c,n_neighbors=50,n_jobs=-1)
#    mf = pca.fit_transform(MF)

n_total = Score.shape[0]
Result = np.zeros(n_total)
Result[np.where(Score[:, 0] > Score[:, 1])] = 1
Result[np.where(Score[:, 0] < Score[:, 1])] = -1

classif = RFC(n_estimators=1000, n_jobs=-1)
u = rd.permutation(n_total)
n_train = int(2 * n_total / 3)

X_train = mf[u[:n_train], :]
X_test = mf[u[n_train:], :]
classif.fit(X_train, Result[u[:n_train]])
Pred = classif.predict(X_test)
print(np.sum(Pred == Result[u[n_train:]]) / (n_total - n_train))
L.append(np.sum(Pred == Result[u[n_train:]]) / (n_total - n_train))

plt.figure()
plt.plot(L, 'k+')
plt.show()
