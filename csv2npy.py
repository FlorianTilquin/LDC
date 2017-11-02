#! /usr/bin/env python3
# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

df = pd.read_csv("ligue1_conforama.csv")

match_day = df.stage # A quelle daynée à lieu le match
date = df.date # la date du match
home_team_id = df.home_team_api_id.as_matrix()
away_team_id = df.away_team_api_id.as_matrix()
home_goals = df.home_team_goal.as_matrix()
away_goals = df.away_team_goal.as_matrix()

# let's parse a little
year = []
month = []
day = []
for d in date:
    dtime = datetime.strptime(d[:10],'%Y-%m-%d')
    year.append(dtime.year)
    month.append(dtime.month)
    day.append(dtime.day)

day   = np.array(day) # que des int, faire gaffe
month = np.array(month)
year  = np.array(year)

print(len(day))
# print(len(np.unique(home_team_id)))
# Score : matrice 3d (nbre_équipe, nbre_matchs, nbre_matchs)
#               id_équipe, but_mis, but_pris)
# Score = np.zeros((
