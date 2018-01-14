#! /usr/bin/env python3
# -*- encoding:utf-8 -*-


import sqlite3 as lite
from time import time as ti

import numpy as np

import pandas as pd

con = lite.connect('database.sqlite')
players = pd.read_sql_query('select * from Player_attributes', con)
#  players = pd.read_csv('players.csv')


def parse_date(date):
    return pd.to_numeric(pd.to_datetime(date)).as_matrix()


def get_player_feat(player_id, corresp, date):
    # le 33 est hardcodé pour repérer direct les problèmes de dim, mais ça reste moche
    if np.isnan(player_id):
        return np.array([np.NaN] * 33).reshape(1, -1).astype(float)
    else:
        player = corresp[str(int(player_id))]
        date = parse_date(date)
        play_date = parse_date(player.date)
        d0 = np.argmin(np.abs(date - play_date))
        play_feat = player.iloc[d0]
        return play_feat['crossing':'gk_reflexes'].as_matrix().reshape(1, 33).astype(float)


def get_team_feat(player_list_id, date, corresp, meaned=True):
    goal_feat = get_player_feat(player_list_id[0], corresp, date)[
        0, -5:]  # Gardien à part
    if meaned:
        try:
            team_feat = np.vstack([get_player_feat(Id, corresp, date)[0, :-5]
                                   for Id in player_list_id[1:]])
        except ValueError:
            print([get_player_feat(Id, corresp, date).shape for Id in player_list_id[1:]])
        team_feat = np.nanmean(team_feat, 0)
    else:
        team_feat = np.hstack([get_player_feat(Id, corresp, date)[0, :-5]
                               for Id in player_list_id[1:]])
    return np.hstack((goal_feat, team_feat))
