#! /usr/bin/env python3
# -*- encoding:utf-8 -*-
#

import pickle as pkl
import sqlite3 as lite

import numpy as np

import pandas as pd
from players_features import get_team_feat

con = lite.connect('database.sqlite')
df = pd.read_sql_query('select * from match', con)
League = df.league_id
leagues = np.unique(League.as_matrix())
n_league = len(leagues)
dfleague = df.groupby('league_id')

MEANED = False  # False va produire des erreurs à cause des Nan, à moins qu'on considère un algo
# qui gère les missings values, genre EM


def mean_up(ex, n_m, goals):
    # n_m -> nbre match home/away inclus
    return (ex * (n_m - 1) + goals) / n_m


def diff_to_pt(d):
    return 3 * (d > 0) if d != 0 else 1


def id_ft(ft):
    return team_ft.index(ft)


Score = {}
M_ft = {}
for league in leagues:
    dfl = dfleague.get_group(league)
    dfl.sort_values(by='date', inplace=True)
    dfl.reset_index(inplace=True)

    Score[str(league)] = dfl.loc[:, 'home_team_goal':'away_team_goal'].as_matrix()
    BM = dfl.loc[:, 'B365H':'BSA'].as_matrix()
    BM.shape
    BM = BM.reshape(-1, 10, 3)

    teams = np.unique(dfl.home_team_api_id).tolist()
    n_team = len(teams)
    dates = np.unique(dfl.date).tolist()
    n_date = len(dates)
    seasons = np.unique(dfl.season)
    n_season = len(seasons)

    team_ft = ['gfh', 'gah', 'n_h', 'gfa', 'gaa', 'n_a', 'm_pt']
    n_ft = len(team_ft)
    Team_ft = np.zeros((n_date, n_team, n_ft))
    Match_ft = np.zeros((len(dfl), 2 * (n_ft + 5 + 28 * (10 - 9 * MEANED)) + 30))

    prev_seas = dfl.season[0]
    k = 0
    for i in range(len(dfl)):
        date = dates.index(dfl.date[i])
        print("features for day ", dfl.date[i])
        home_team = teams.index(dfl.home_team_api_id[i])
        away_team = teams.index(dfl.away_team_api_id[i])
        cur_seas = dfl.season[i]
        if cur_seas != prev_seas:
            k = 0
        if k < len(np.unique(dfl.loc[dfl.season == dfl.season[0], 'home_team_api_id'])) / 2:
            Team_ft[date:, home_team, :] = 0  # Ça devrait fonctionner a priori ?
            # Les 10 premiers matchs de la saison correspondent forcément à la 1re journée de L1
            Team_ft[date:, away_team, :] = 0
            k = k + 1
        prev_seas = cur_seas

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
            Match_ft[i, n_ft:2 * n_ft] = Team_ft[date - 1, away_team, :]
            # Match_ft[i, 2 * n_ft:] = BM[i, ...].reshape(1, -1)

        #  Add players features (33 per player, thus 5+28*10 if MEANED=False else 33) to the list
        player_list_id = dfl.loc[i, 'home_player_1':'away_player_11'].as_matrix()
        ht_feat = get_team_feat(player_list_id[:11], pd.Series(dfl.date[i]), MEANED)
        at_feat = get_team_feat(player_list_id[11:], pd.Series(dfl.date[i]), MEANED)
        Match_ft[i, 2 * n_ft:2 * n_ft + 5 + 28 * (10 - 9 * MEANED)] = ht_feat
        Match_ft[i, 2 * n_ft + 5 + 28 * (10 - 9 * MEANED):2 *
                 (n_ft + 5 + 28 * (10 - 9 * MEANED))] = at_feat

        #  Add Bookmaker features
        bm = BM[i, ...]
        bm[np.any(np.isnan(bm), 1), :] = 1 / np.array([0.4587, 0.2539, 0.2874])
        Match_ft[i, -30:] = bm.reshape(1, -1)

    M_ft[str(league)] = Match_ft

pkl.dump(M_ft, open('M_ft.p', 'wb'))
pkl.dump(Score, open('Score.p', 'wb'))
# np.save('Score.npy', Score)
# np.save('Team_features.npy', Team_ft)
# print(Match_ft.shape)
# np.save('Match_features.npy', Match_ft)
