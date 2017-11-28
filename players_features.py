
#! /usr/bin/env python3
# -*- encoding:utf-8 -*-


import numpy as np

import pandas as pd

players = pd.read_csv('players.csv')


def parse_date(date):
    return pd.to_numeric(pd.to_datetime(date)).as_matrix()


def get_player_feat(player_id, date):
    if np.isnan(player_id):
        return np.array([np.NaN] * 33).reshape(1, -1).astype(float)
    else:
        player = players.loc[players.player_api_id == player_id]
        date = parse_date(date)
        play_date = parse_date(player.date_stat)
        d0 = np.argmin(np.abs(date - play_date))
        play_feat = player.iloc[d0]
        return play_feat[10:].as_matrix().reshape(1, -1).astype(float)


def get_team_feat(player_list_id, date, meaned=True):
    goal_feat = get_player_feat(player_list_id[0], date)[0, -5:]
    if meaned:
        try :
            team_feat = np.vstack([get_player_feat(Id, date)[0, :-5] for Id in player_list_id[1:]])
        except:
            print([get_player_feat(Id, date).shape for Id in player_list_id[1:]])
        team_feat = np.nanmean(team_feat, 0)
    else:
        team_feat = np.hstack([get_player_feat(Id, date)[0, :-5] for Id in player_list_id[1:]])
    return np.hstack((goal_feat, team_feat))