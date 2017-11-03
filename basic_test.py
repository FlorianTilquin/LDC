#! /usr/bin/env python3
# -*- encoding:utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR

M = np.load('ASFM.npy')
P = M.copy()

for j in range(8):
    for i in range(20):
        P[j,i,:,:2] = np.cumsum(M[j,i,:,:2],0)/(1+np.arange(38)[:,None]) # on normalise les buts mis/pris par matchs joués dans la saison

F = P[...,:3].reshape(-1,3)
S = M[...,:2].reshape(-1,2)

# On essaie de prédire le score des matchs avec pour seul features les buts/match de l'équipe qui accueille et dom/ext
n_tr = 38*20*7
clf = RFR(n_estimators = 1000,n_jobs=-1)
clf.fit(F[:n_tr],S[:n_tr]) # on entraîne sur les 7 premières saisons
S_p = clf.predict(F[n_tr:])
print(np.mean(np.abs(S_p-S[n_tr:]),0))
print(clf.feature_importances_)
