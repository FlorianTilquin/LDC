
# coding: utf-8

# # European Soccer

# In[1]:


import collections
import csv
import numpy as np
import pandas as pd
import sqlite3 as lite

from sklearn.preprocessing import normalize


# In[101]:


pd.options.mode.chained_assignment = None  # default='warn'


# In[3]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## 1. Extracting Data
# dataset available at: https://www.kaggle.com/hugomathien/soccer/data
# 
# complement available at: https://www.kaggle.com/jiezi2004/soccer

# In[338]:


con = lite.connect('database.sqlite')
df = pd.read_sql_query('select * from match', con)
players = pd.read_sql_query('select * from player_Attributes', con)


# In[339]:


df.sort_values(by='date', inplace=True, ascending=False)
players.sort_values(by='date', inplace=True, ascending=False)


# In[340]:


df.reset_index(inplace=True)
players.reset_index(inplace=True)


# ## 2. Modifying the tables to take usable values

# In[313]:


"""
maps all the players attributes to floats between 0. and 1.

the attribute 'preffered_foot' is mapped to [1., 0.] or [0., 1.]

values 'low', 'medium' and 'high' are mapped to 0., 0.5 and 1.
"""
def map_attributes(df):
    df['other_foot'] = df['preferred_foot'].copy()
    
    val1 = df.loc[:, 'preferred_foot'].map(lambda x: 100. - 100. * (x == 'left'))
    val2 = df.loc[:, 'preferred_foot'].map(lambda x: 0. + 100. * (x == 'left'))
    val3 = df.loc[:,'attacking_work_rate':'defensive_work_rate'].applymap(lambda x: 0. + 50.*(x=='medium') + 100.*(x=='high'))
    
    df.loc[:, 'other_foot'] = val1
    df.loc[:, 'preferred_foot'] = val2
    df.loc[:,'attacking_work_rate':'defensive_work_rate'] = val3
   
    df.loc[:, 'overall_rating':] = df.loc[:, 'overall_rating':].applymap(lambda x: x/100.)
    
    return df


# In[341]:


player_attributes = map_attributes(players)


# player_attributes.head()

# In[342]:


mean_attributes = player_attributes.loc[:, 'overall_rating':].mean()


# mean_attributes

# In[343]:


mat_mean_attributes = mean_attributes.as_matrix()


# ## 3. Extracting desired features

# ### 3.1 Match features

# In[344]:


"""
assumes that df is sorted by dates in decreasing order

returns the last 'n_matchs' matchs of team 'team_id' anterior to 'date'

location can be 'home', 'away' or None (which means indifferent here)
"""
def last_matchs(df, team_id, date, n_matchs, location = None):
    if location == 'home':
        df = df[df['home_team_api_id']==team_id]
    if location == 'away':
        df = df[df['away_team_api_id']==team_id]
    else:
        df = pd.concat([df[df['home_team_api_id']==team_id],df[df['away_team_api_id']==team_id]])
    
    df = df[df['date'] < date]
    #df.sort_values(by='date', ascending=False, inplace=True)
    df.reset_index(inplace=True)

    n_matchs = min(n_matchs, len(dum_df)) - 1
    
    return df.loc[0:n_matchs]


# dum = last_matchs(df, 9987, '2017-12-30', 5, 'away')

# dum

# In[345]:


def match_features(df):
    df = df.loc[:, 'home_team_goal':'away_team_goal']
    
    return df.as_matrix().reshape(-1)


# match_features(dum)

# ### 3.2 Team features

# In[346]:


"""
assumes that the table of players attributes is sorted in decreasing order

returns the most recent attributes for player 'player_id' up to date 'date'.
"""
def player_features(df, player_id, date):    
    df = df[df['player_api_id']==player_id]
    
    df = df[df['date'] <= date]
    #df.sort_values(by='date', ascending=False, inplace=True)
    df.reset_index(inplace=True)
    
    if 0 in df.index:
        res = df.loc[0,'overall_rating':].as_matrix()
    else:
        res = mat_mean_attributes
        
    res[pd.isnull(res)] = mat_mean_attributes[pd.isnull(res)]
        
    return res


# player_features(player_attributes, 505942, '2017-01-01')

# In[347]:


"""
Y donne la position de but (1) à rond central (11)
X donne la position de gauche (1) à droite (9)
"""
def team_features(df_row, attributes, location = 'home'):
        date = df_row['date']
        ids = df_row[location+'_player_1':location+'_player_11'].as_matrix()
        X = df_row[location+'_player_X1':location+'_player_X11'].as_matrix()
        Y = df_row[location+'_player_Y1':location+'_player_Y11'].as_matrix()
        
        positions = {i : (ids[i],X[i],Y[i]) for i in range(11)}
        positions = sorted(positions.items(), key=lambda x: (x[1][2], x[1][1]))

        res = []
        for i, val in positions:
            player = val[0]
            res += list(player_features(attributes, player, date))
            
        return res


# team_features(df.loc[0], player_attributes)

# ### 3.3 Bookmakers features

# In[348]:


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


# BM_features(df.loc[0])

# ### 3.4 Concatenation

# In[349]:


def all_features(df_matchs, df_players, n_matchs):
    features = []
    ground_truth = []
    
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
        
        #matchs features
        home_team_last_matchs_home = match_features(last_matchs(df, home_team_id, date, n_matchs, location = 'home'))
        home_team_last_matchs_away = match_features(last_matchs(df, home_team_id, date, n_matchs, location = 'away'))
        away_team_last_matchs_home = match_features(last_matchs(df, away_team_id, date, n_matchs, location = 'home'))
        away_team_last_matchs_away = match_features(last_matchs(df, away_team_id, date, n_matchs, location = 'away'))
        
        matchs_feat = np.zeros(4 * 2 * n_matchs)
        
        matchs_feat[0 * n_matchs : 0 * n_matchs+len(home_team_last_matchs_home)] += home_team_last_matchs_home
        matchs_feat[2 * n_matchs : 2 * n_matchs+len(home_team_last_matchs_away)] += home_team_last_matchs_away
        matchs_feat[4 * n_matchs : 4 * n_matchs+len(away_team_last_matchs_home)] += away_team_last_matchs_home
        matchs_feat[6 * n_matchs : 6 * n_matchs+len(away_team_last_matchs_away)] += away_team_last_matchs_away
        
        #team features
        home_team_feat = team_features(row, df_players, location = 'home')
        away_team_feat = team_features(row, df_players, location = 'away')
        
        team_feat = np.array(home_team_feat + away_team_feat)
        
        if np.count_nonzero(np.isnan(team_feat)) > 0:
            return team_feat
        
        #bookmakers features
        bm_feat = BM_features(row)
        
        #concatenation
        all_feat = np.concatenate((matchs_feat, team_feat, bm_feat))
        gt = 1 + np.sign(row['home_team_goal'] - row['away_team_goal'])
        
        features.append(all_feat)
        ground_truth.append(gt)
        
    return features, ground_truth


# In[336]:


feat, GT = all_features(df, player_attributes, 10)


# In[210]:


import pickle


# In[272]:


pickle.dump(feat, open('new_features.p', 'wb'))


# In[215]:


pickle.dump(GT, open('ground_truth.p', 'wb'))

