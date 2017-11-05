#! /usr/bin/env python3
# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
import sqlite3

# df = pd.read_csv("ligue1_conforama.csv") ## SO OLD SCHOOL
con = sqlite3.connect('database.sqlite')
df = pd.read_sql_query('select * from match where league_id=4679',con) #league_id 4679  = ligue1 conforama
df.sort_values(by='date',inplace=True)

match_day = df.stage # A quelle journée a lieu le match
date = df.date # la date du match
season = df.season.as_matrix()
ligue_stage = df.stage.as_matrix()
home_team_id = df.home_team_api_id.as_matrix()
away_team_id = df.away_team_api_id.as_matrix()
home_goals = df.home_team_goal.as_matrix()
away_goals = df.away_team_goal.as_matrix()

## Pour l'instant on veut juste l'ordre des matchs: (pour les journées de championnat sur plusieurs jours, ou les matchs reportés pour coup ou autre)
# order = sorted(range(len(date)), key = lambda x: dt.strptime(date[x][:10],'%Y-%m-%d')) # En fait pandas c'est trop bien et ça le fait pour nous

# L = [date,season,ligue_stage,home_team_id,away_team_id,home_goals,away_goals]
# date,season,ligue_stage,home_team_id,away_team_id,home_goals,away_goals = (l[order] for l in L) #360 no scope


# M : feature matrice 4d (nbre_saisons=8,nbre_équipe=20, nbre_matchs=38, nbre_ft=4)
              # but_mis, but_pris, domicile/ext, adversaire, equipe_id)
M = np.zeros((8,20,38,5),dtype=np.float64)
Teams = np.zeros((8,20)) # Les teams id, pour savoir qui affronte qui
for i,s in enumerate(np.unique(season)):
    dum = (season == s) #on reduit un peu les calculs au cas où...
    htid = home_team_id[dum]
    atid = away_team_id[dum]
    hg = home_goals[dum]
    ag = away_goals[dum]
    teams = np.unique(htid)
    Teams[i,:] = teams
    for j,eq in enumerate(teams):
        matchs_dom = (htid==eq)
        matchs_ext = (atid==eq)
        mce = np.where(matchs_dom|matchs_ext) #matchs contenant l'equipe
        hg_mce = hg[mce][:,None]
        ag_mce = ag[mce][:,None]
        scores_mce = np.hstack((hg_mce,ag_mce))
        tid_mce = np.vstack((htid[mce],atid[mce])).T
        dom_ext = (htid[mce]==eq).astype(int)#si match à domicile = 1
        M[i,j,:,0] = scores_mce[range(38),dom_ext]#but mis
        M[i,j,:,1] = scores_mce[range(38),1-dom_ext]#but pris
        M[i,j,:,2] = dom_ext
        M[i,j,:,3] = tid_mce[range(38),dom_ext]
        M[i,j,:,4] = eq

np.save('Teams',Teams)
np.save('ASFM',M) # all seasons feature matrix
