
# coding: utf-8

# # European Soccer

# In[1]:


import collections
import csv
import numpy as np
import pandas as pd
import sqlite3 as lite

from sklearn.preprocessing import normalize
from sklearn.manifold import spectral_embedding as SE


# In[101]:


pd.options.mode.chained_assignment = None  # default='warn'


# In[3]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## 1. Extracting Data
# dataset available at: https://www.kaggle.com/hugomathien/soccer/data
#
# complement available at: https://www.kaggle.com/jiezi2004/soccer

# In[407]:


con = lite.connect('database.sqlite')
df = pd.read_sql_query('select * from match', con)
players = pd.read_sql_query('select * from player_Attributes', con)


# ## 2. Modifying the tables to take usable values

# In[391]:


"""
maps all the players attributes to floats between 0. and 1.

the attribute 'preffered_foot' is mapped to [1., 0.] or [0., 1.]

values 'low', 'medium' and 'high' are mapped to 0., 0.5 and 1.
"""
def map_attributes(attributes):
    df = attributes.copy()
    df['other_foot'] = df['preferred_foot'].copy()

    val1 = df.loc[:, 'preferred_foot'].map(lambda x: 100. - 100. * (x == 'left'))
    val2 = df.loc[:, 'preferred_foot'].map(lambda x: 0. + 100. * (x == 'left'))
    val3 = df.loc[:,'attacking_work_rate':'defensive_work_rate'].applymap(lambda x: 0. + 50.*(x=='medium') + 100.*(x=='high'))

    df.loc[:, 'other_foot'] = val1
    df.loc[:, 'preferred_foot'] = val2
    df.loc[:,'attacking_work_rate':'defensive_work_rate'] = val3

    df.loc[:, 'overall_rating':] = df.loc[:, 'overall_rating':].applymap(lambda x: x/100.)

    return df


# ## 3. Extracting desired features

# ### 3.0 Power ranking features
def power_ranking_features(M , n_feat=5):
    pwr = SE(M,n_feat)
    return pwr

# ### 3.1 Match features

# In[392]:


"""
returns a dictionnary with teams as keys and df as values containing all the matches played by the key team at 'location'
"""
def matchs_by_team(df, location = 'home'):
    res = {}
    teams = df[location+'_team_api_id'].unique()

    for team in teams:
        res[team] = df[df[location+'_team_api_id']==team]

    return res


# In[405]:


"""
assumes that df is sorted by dates in decreasing order

returns the last 'n_matchs' matchs of team 'team_id' anterior to 'date'

location can be 'home', 'away' or None (which means indifferent here)
"""
def last_matchs(df, date, n_matchs):

    dum_df = df[df['date'] < date]
    dum_df.reset_index(inplace=True, drop=True)

    n_matchs = min(n_matchs, len(dum_df)) - 1

    return dum_df.loc[0:n_matchs]


# In[394]:


def match_features(df):
    return (df.loc[:, 'home_team_goal':'away_team_goal']).as_matrix().reshape(-1)


# ### 3.2 Team features

# In[395]:


def attributes_by_player(attributes):
    res = {}
    players = attributes['player_api_id'].unique()

    for player in players:
        res[player] = attributes[attributes['player_api_id']==player]

    return res

# In[401]:


"""
assumes that the table of players attributes is sorted in decreasing order

returns the most recent attributes for player 'player_id' up to date 'date'.
"""
def player_features(player_attributes, date, mat_mean_attributes):

    df = player_attributes[player_attributes['date'] <= date]
    df.reset_index(inplace=True, drop=True)

    if 0 in df.index:
        res = df.loc[0,'overall_rating':].as_matrix()
    else:
        res = mat_mean_attributes

    res[pd.isnull(res)] = mat_mean_attributes[pd.isnull(res)]

    return res


# In[397]:


"""
Y donne la position de but (1) à rond central (11)
X donne la position de gauche (1) à droite (9)
"""
def team_features(df_row, attributes, mat_mean_attributes, location = 'home'):
        date = df_row['date']
        ids = df_row[location+'_player_1':location+'_player_11'].as_matrix()
        X = df_row[location+'_player_X1':location+'_player_X11'].as_matrix()
        Y = df_row[location+'_player_Y1':location+'_player_Y11'].as_matrix()

        positions = {i : (ids[i],X[i],Y[i]) for i in range(11)}
        positions = sorted(positions.items(), key=lambda x: (x[1][2], x[1][1]))

        res = []
        for i, val in positions:
            player = val[0]
            if player in attributes:
                res += list(player_features(attributes[player], date, mat_mean_attributes))
            else:
                res += list(mat_mean_attributes)

        return res


# ### 3.3 Bookmakers features

# In[398]:


def avg_val(s):
    return 0.46 * (s[-1] == 'H') + 0.29 * (s[-1] == 'D') + 0.25 * (s[-1] == 'A')

def BM_features(row):
    row = row.loc['B365H':'BSA']
    row = row.map(lambda x: 1/x)

    values = {key: avg_val(key) for key in row.index}
    row.fillna(value=values, inplace=True)

    bm = (row.as_matrix()).reshape(-1, 10, 3)
    for i in range(len(bm)):
        bm[i] = normalize(bm[i], axis = 1, norm = 'l1')

    return bm.reshape(-1)


# ### 3.4 Concatenation

# In[403]:


def all_features(df_matchs, df_players, n_matchs):
    df_matchs.sort_values(by='date', inplace=True, ascending=False)
    df_players.sort_values(by='date', inplace=True, ascending=False)

    df_matchs.reset_index(inplace=True, drop=True)
    df_players.reset_index(inplace=True, drop=True)

    home_matchs_by_team = matchs_by_team(df, location = 'home')
    away_matchs_by_team = matchs_by_team(df, location = 'away')

    df_players = map_attributes(df_players)
    att_by_player = attributes_by_player(df_players)
    mat_mean_attribute = (df_players.loc[:, 'overall_rating':].mean()).as_matrix()

    features = []
    ground_truth = []

    league_matrices = {}
    league_team_ids = {}

    for row in df_matchs.iterrows():
        i = row[0]
        row = row[1]


        if i%100 == 99:
            print('.', sep=' ', end='', flush=True)
        if i%1000 == 999:
            print(i+1)

        home_team_id = row['home_team_api_id']
        away_team_id = row['away_team_api_id']
        date = row['date']

        #Power Ranking features
        lid = row.league_id
        if  not in league_list :
            league = df.loc[df.league_id==lid]
            n_team = len(np.unique(league.home_team_api_id))
            league_matrices[str(lid)] = np.zeros((n_team,)*2)
            league_team_id[str(lid)] = np.unique(league.home_team_api_id)
        else :
            M = league_matrices[str(lid)] #Passage par ref ici,
            E = league_team_id[str(lid)]
            i = np.where(E==home_team_id)[0][0]
            j = np.where(E==away_team_id)[0][0]
            M[i,j] +=  row['home_team_goal']
            M[j,i] +=  row['away_team_goal']
        prf_feat = get_power_ranking_features(M)
        prf_feat = prf_feat[[i,j],:]

        #matchs features
        home_team_last_matchs_home = match_features(last_matchs(home_matchs_by_team[home_team_id], date, n_matchs))
        home_team_last_matchs_away = match_features(last_matchs(away_matchs_by_team[home_team_id], date, n_matchs))
        away_team_last_matchs_home = match_features(last_matchs(home_matchs_by_team[away_team_id], date, n_matchs))
        away_team_last_matchs_away = match_features(last_matchs(away_matchs_by_team[away_team_id], date, n_matchs))

        matchs_feat = np.zeros(4 * 2 * n_matchs)

        matchs_feat[0 * n_matchs : 0 * n_matchs+len(home_team_last_matchs_home)] += home_team_last_matchs_home
        matchs_feat[2 * n_matchs : 2 * n_matchs+len(home_team_last_matchs_away)] += home_team_last_matchs_away
        matchs_feat[4 * n_matchs : 4 * n_matchs+len(away_team_last_matchs_home)] += away_team_last_matchs_home
        matchs_feat[6 * n_matchs : 6 * n_matchs+len(away_team_last_matchs_away)] += away_team_last_matchs_away

        #team features
        home_team_feat = team_features(row, att_by_player, mat_mean_attributes, location = 'home')
        away_team_feat = team_features(row, att_by_player, mat_mean_attributes, location = 'away')

        team_feat = np.array(home_team_feat + away_team_feat)

        #bookmakers features
        bm_feat = BM_features(row)

        #concatenation
        all_feat = np.concatenate((matchs_feat, team_feat, bm_feat, prf_feat))
        gt = 1 + np.sign(row['home_team_goal'] - row['away_team_goal'])

        features.append(all_feat)
        ground_truth.append(gt)

    return features, ground_truth


# In[406]:


feat, GT = all_features(df, players, 10)


# In[210]:


import pickle


# In[272]:


pickle.dump(feat, open('new_features.p', 'wb'))


# In[215]:


pickle.dump(GT, open('ground_truth.p', 'wb'))

