#! /usr/bin/env python3
# -*- encoding:utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC

M = np.load('ASFM.npy')
Teams = np.load('Teams.npy')
P = M.copy()
adv = np.zeros((8,20,38),dtype=int)

for j in range(8):
    teams = Teams[j].astype(int)
    dum = np.zeros(np.max(teams)+1,dtype=int) # c'est très laid, mais j'ai la flemme de faire plus joli pour l'instant
    dum[teams] = np.arange(20)
    adv[j,:,:] = dum[P[j,:,:,-1].astype(int)]
    for i in range(20):
        P[j,i,:,:2] = np.cumsum(M[j,i,:,:2],0)/(1+np.arange(38)[:,None]) # on normalise les buts mis/pris par matchs joués dans la saison

F_adv = np.zeros((8,20,38,2)) #les "features" des adversaires, i.e. but mis/pris par match
for k in range(8):
    for l in range(20):
        F_adv[k,l,:,:] = P[j,k,adv[k,l],:2]

F = P[...,:3].reshape(-1,3)
F_adv = F_adv.reshape(-1,2)
# F = np.hstack((F,F_adv))
S = M[...,:2].reshape(-1,2)

sco = np.zeros(8*38*20)
sco[np.where(S[:,0]>S[:,1])] = 1 #home victory
sco[np.where(S[:,0]<S[:,1])] = -1 #home loss

# On essaie de prédire le score des matchs avec pour seul features les buts/match de l'équipe qui accueille et dom/ext
n_tr = 38*20*7

# normalization ? Useless pour RF
# F = F-np.mean(F,0)[None,:]
# F /= np.std(F,0)[None,:]

# Score prediction
clf = RFR(n_estimators = 500,n_jobs=-1)
clf.fit(F[:n_tr],S[:n_tr]) # on entraîne sur les 7 premières saisons
S_p = clf.predict(F[n_tr:])
print(np.mean(np.abs(S_p-S[n_tr:]),0))
print(clf.feature_importances_)

# Result prediction
clf = RFC(n_estimators = 500,class_weight='balanced',n_jobs=-1)
clf.fit(F[:n_tr],sco[:n_tr]) # on entraîne sur les 7 premières saisons
S_p = clf.predict(F[n_tr:])
print(np.mean(S_p == sco[n_tr:]))
