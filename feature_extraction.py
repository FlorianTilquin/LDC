#! /usr/bin/env python3
# -*- encoding:utf-8 -*-
#

import sqlite3 as lite

import numpy as np

import pandas as pd
from players_features import get_team_feat

con = lite.connect('database.sqlite')
df = pd.read_sql_query('select * from match where league_id = 4735', con)
df.sort_values(by='date', inplace=True)
df.reset_index(inplace=True)

Score = df.loc[:,'home_team_goal':'away_team_goal'].as_matrix()
np.save('Score.npy',Score)

teams = np.unique(df.home_team_api_id).tolist()
n_team = len(teams)
dates = np.unique(df.date).tolist()
n_date = len(dates)

team_ft = ['gfh', 'gah', 'n_h', 'gfa', 'gaa', 'n_a', 'm_pt']
n_ft = len(team_ft)
Team_ft = np.zeros((n_date, n_team, n_ft))
MEANED = True  # False va produire des erreurs à cause des Nan, à moins qu'on considère un algo qui
# gère les missings values, genre EM
Match_ft = np.zeros((len(df), 2 * (n_ft + 5 + 28 * (11 - 10 * MEANED))))
# BM = np.load('Probas_BM.npy')
# Remplissage de la matrice de ft. par equipe et par date unique
# Match_ft : [home_ft,away_ft,BM_ft]


def mean_up(ex, n_m, goals):
    # n_m -> nbre match home/away inclus
    return (ex * (n_m - 1) + goals) / n_m


def diff_to_pt(d):
    return 3 * (d > 0) if d != 0 else 1


def id_ft(ft):
    return team_ft.index(ft)


for i in range(len(df)):
    date = dates.index(df.date[i])
    print("features for day ", df.date[i])
    home_team = teams.index(df.home_team_api_id[i])
    away_team = teams.index(df.away_team_api_id[i])
    if i % 380 < 10:
        Team_ft[date:, home_team, :] = 0  # Ça devrait fonctionner a priori ?
        # Les 10 premiers matchs de la saison correspondent forcément à la première journée de L1
        Team_ft[date:, away_team, :] = 0

    htg = df.home_team_goal[i]
    atg = df.away_team_goal[i]
    dtg = htg - atg

    # HT Features update
    m_p = Team_ft[date, home_team, id_ft('m_pt')]
    pts = diff_to_pt(dtg)
    n_home = Team_ft[date, home_team, id_ft('n_h')] + 1
    Team_ft[date:, home_team, id_ft('n_h')] = n_home

    n_match = Team_ft[date, home_team, id_ft(
        'n_h')] + Team_ft[date, home_team, id_ft('n_a')]
    Team_ft[date:, home_team, id_ft('m_pt')] = mean_up(m_p, n_match, pts)

    gfh = Team_ft[date, home_team, id_ft('gfh')]
    gah = Team_ft[date, home_team, id_ft('gah')]
    Team_ft[date:, home_team, id_ft('gfh')] = mean_up(gfh, n_home, htg)
    Team_ft[date:, home_team, id_ft('gah')] = mean_up(gah, n_home, atg)

    # AT Features update
    m_p = Team_ft[date, away_team, id_ft('m_pt')]
    pts = diff_to_pt(dtg)
    n_away = Team_ft[date, away_team, id_ft('n_a')] + 1
    Team_ft[date:, away_team, id_ft('n_a')] = n_away

    n_match = Team_ft[date, away_team, id_ft(
        'n_h')] + Team_ft[date, away_team, id_ft('n_a')]
    Team_ft[date:, away_team, id_ft('m_pt')] = mean_up(m_p, n_match, pts)

    gfa = Team_ft[date, away_team, id_ft('gfa')]
    gaa = Team_ft[date, away_team, id_ft('gaa')]
    Team_ft[date:, away_team, id_ft('gfa')] = mean_up(gfa, n_away, atg)
    Team_ft[date:, away_team, id_ft('gaa')] = mean_up(gaa, n_away, htg)

    # Match ft. filling
    if date > 0:
        Match_ft[i, :n_ft] = Team_ft[date - 1, home_team, :]
        if i % 380 == 1:
            print('b', np.sum(Team_ft[date - 1, home_team, :]))
        Match_ft[i, n_ft:2 * n_ft] = Team_ft[date - 1, away_team, :]
        # Match_ft[i, 2 * n_ft:] = BM[i, ...].reshape(1, -1)

    #  Add players features (33 per player, thus 5+28*11 if MEANED=False else 33) to the list
    player_list_id = df.loc[i, 'home_player_1':'away_player_11'].as_matrix()
    ht_feat = get_team_feat(player_list_id[:11], pd.Series(df.date[i]), MEANED)
    at_feat = get_team_feat(player_list_id[11:], pd.Series(df.date[i]), MEANED)
    Match_ft[i,2 * n_ft:2 * n_ft + 5 + 28 * (11 - 10 * MEANED)] = ht_feat
    Match_ft[i,2 * n_ft + 5 + 28 * (11 - 10 * MEANED):2 *
             (n_ft + 5 + 28 * (11 - 10 * MEANED))] = at_feat
np.save('Team_features.npy', Team_ft)
np.save('Match_features.npy', Match_ft)
