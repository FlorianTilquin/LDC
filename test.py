#! /usr/bin/env python3
# -*- encoding:utf-8 -*-

import pickle as pkl

import numpy as np
import numpy.random as rd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR

# from sklearn.manifold import Isomap

Score = pkl.load(open("Score.p", "rb"))  # np.load('Score.npy')
MF = pkl.load(open("M_ft.p", "rb"))  # np.load('Match_features.npy')
Score = np.vstack([Score[key] for key in Score.keys()])
MF = np.vstack([MF[key] for key in MF.keys()])
MF[np.isnan(MF)] = np.nanmean(MF)
print(MF.shape)
L = []
mf = MF.copy()
pca = PCA()
pca.fit(mf)
plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.show()

n_total = Score.shape[0]
Result = np.zeros(n_total)
Result[np.where(Score[:, 0] > Score[:, 1])] = 1
Result[np.where(Score[:, 0] < Score[:, 1])] = -1

u = rd.permutation(n_total)
n_train = int(2 * n_total / 3)

X_train = mf[u[:n_train], :]
X_test = mf[u[n_train:], :]
for n_c in range(1, 80):
    pca = PCA(n_components=n_c)
    # pca = Isomap(n_components=n_c,n_neighbors=50,n_jobs=-1)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    classif = RFC(n_estimators=500, n_jobs=-1)
    classif.fit(X_tr, Result[u[:n_train]])
    Pred = classif.predict(X_te)
    print(np.sum(Pred == Result[u[n_train:]]) / (n_total - n_train))
    L.append(np.sum(Pred == Result[u[n_train:]]) / (n_total - n_train))

plt.figure()
plt.plot(L, 'k+')
plt.show()
