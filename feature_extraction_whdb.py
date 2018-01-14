#! /usr/bin/env python3
# -*- encoding:utf-8 -*-
#

import pickle as pkl
import sqlite3 as lite
from time import time as ti

import numpy as np

import pandas as pd
from players_features import get_team_feat

con = lite.connect('database.sqlite')
df = pd.read_sql_query('select * from match', con)
League = df.league_id
Leagues = np.unique(League.as_matrix())
n_league = len(Leagues)
dfleague = df.groupby('league_id')

MEANED = False

corresp = pkl.load(open('corresp.pkl', 'rb'))


def mean_up(ex, n_m, goals):
    # n_m -> nbre match home/away inclus
    return (ex * (n_m - 1) + goals) / n_m


def diff_to_pt(d):
    return 3 * (d > 0) if d != 0 else 1


team_ft = ['gfh', 'gah', 'n_h', 'gfa', 'gaa', 'n_a', 'm_pt']
n_ft = len(team_ft)


def id_ft(ft):
    return team_ft.index(ft)


def team_features_update(Team_ft, date, Tid, Goals, erase, HOME):
    htg, atg, pts = Goals
    per = slice(date, date + 100)
    if erase:
        Team_ft[per, Tid, :] = 0
    # for ft in team_ft:
    #     globals()[ft] = Team_ft[date, Tid, id_ft(ft)]
    n_h = Team_ft[date, Tid, id_ft('n_h')]
    gfh = Team_ft[date, Tid, id_ft('gfh')]
    gfa = Team_ft[date, Tid, id_ft('gfa')]
    gaa = Team_ft[date, Tid, id_ft('gaa')]
    gah = Team_ft[date, Tid, id_ft('gah')]
    n_a = Team_ft[date, Tid, id_ft('n_a')]
    m_pt = Team_ft[date, Tid, id_ft('m_pt')]

    if HOME:
        n_h += 1
        Team_ft[per, Tid, id_ft('n_h')] = n_h
        Team_ft[per, Tid, id_ft('gfh')] = mean_up(gfh, n_h, htg)
        Team_ft[per, Tid, id_ft('gah')] = mean_up(gah, n_h, atg)
    else:
        n_a += 1
        Team_ft[per, Tid, id_ft('n_a')] = n_a
        Team_ft[per, Tid, id_ft('gfa')] = mean_up(gfa, n_a, atg)
        Team_ft[per, Tid, id_ft('gaa')] = mean_up(gaa, n_a, htg)
    n_match = n_a + n_h + 1
    Team_ft[per, Tid, id_ft('m_pt')] = mean_up(m_pt, n_match, pts * (2 * HOME - 1))
    return Team_ft


Score = {}
M_ft = {}


def extract_match_features(leagues=Leagues, MEANED=False, dyn_length=20, Book=True):
    for league in leagues:
        dfl = dfleague.get_group(league)
        dfl.sort_values(by='date', inplace=True)
        dfl.reset_index(inplace=True)

        Score[str(league)] = dfl.loc[:,
                                     'home_team_goal':'away_team_goal'].as_matrix()
        if Book:
            BM = dfl.loc[:, 'B365H':'BSA'].as_matrix()
            BM = BM.reshape(-1, 10, 3)

        teams = np.unique(dfl.home_team_api_id).tolist()
        dates = np.unique(dfl.date).tolist()
        seasons = np.unique(dfl.season)

        Team_ft = np.zeros((len(dates), len(teams), n_ft))
        Match_ft = np.zeros((len(dfl), 2 * (n_ft + 5 + 28 * (10 - 9 * MEANED)) + 30))

        prev_seas = dfl.season[0]
        k = 0
        for i in range(len(dfl)):
            date = dates.index(dfl.date[i])
            if date == 0:
                continue

            #  Add Bookmaker features
            bm = BM[i, ...]
            bm[np.any(np.isnan(bm), 1), :] = 1 / np.array([0.4587, 0.2539, 0.2874])
            Match_ft[i, -30:] = bm.reshape(1, -1)

            print("features for day ", dfl.date[i], ", league:", league)
            a = ti()
            home_team = teams.index(dfl.home_team_api_id[i])
            away_team = teams.index(dfl.away_team_api_id[i])
            cur_seas = dfl.season[i]
            if cur_seas != prev_seas:
                k = 0
            if k < len(np.unique(dfl.loc[dfl.season == dfl.season[0], 'home_team_api_id'])) / 2:
                erase = True
                k = k + 1
            prev_seas = cur_seas

            # Team Features update
            htg = dfl.home_team_goal[i]
            atg = dfl.away_team_goal[i]
            dtg = htg - atg
            pts = diff_to_pt(dtg)
            Team_ft = team_features_update(Team_ft, date, home_team, [htg, atg, pts], erase, True)
            Team_ft = team_features_update(Team_ft, date, away_team, [htg, atg, pts], erase, False)
            #  TODO FEATURES DYNAMIQUE TODO
            # Home_story = dfl.loc[((dfl.home_team_api_id == home_team) |
            #                       (dfl.away_team_api_id == home_team)) & (dfl.date < dfl.date[i])]
            # gfh_story # là il faut metre à part les matchs où l'équipe était à domicile
            # Away_story = dfl.loc[((dfl.home_team_api_id == away_team) |
            #                       (dfl.away_team_api_id == away_team)) & (dfl.date < dfl.date[i])]

            # Match ft. filling
            Match_ft[i, :n_ft] = Team_ft[date - 1, home_team, :]
            Match_ft[i, n_ft:2 * n_ft] = Team_ft[date - 1, away_team, :]

            #  Add players features (33 per player: 5GK + 28 field) to the list
            player_list_id = dfl.loc[i,
                                     'home_player_1':'away_player_11'].as_matrix()
            ht_feat = get_team_feat(
                player_list_id[:11], pd.Series(dfl.date[i]), corresp, MEANED)
            at_feat = get_team_feat(
                player_list_id[11:], pd.Series(dfl.date[i]), corresp, MEANED)
            Match_ft[i, 2 * n_ft:2 * n_ft + 5 + 28 * (10 - 9 * MEANED)] = ht_feat
            Match_ft[i, 2 * n_ft + 5 + 28 * (10 - 9 * MEANED):2 *
                     (n_ft + 5 + 28 * (10 - 9 * MEANED))] = at_feat

        M_ft[str(league)] = Match_ft

    return M_ft, Score


M_ft, Score = extract_match_features()
# pkl.dump(M_ft, open('M_ft.p', 'wb'))
# pkl.dump(Score, open('Score.p', 'wb'))
