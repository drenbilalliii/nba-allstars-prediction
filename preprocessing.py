import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def preprocessing(df):
    df.drop(columns=['Unnamed: 0', 'blank2', 'blanl'], axis=1, inplace=True)
    name_changes = {"NOK": "NOP", "NOH": "NOP", "SEA": "OKC", "NJN": "BRK", "CHA": "CHO"}
    for key, value in name_changes.items():
        df.loc[df["Tm"] == key, "Tm"] = value
    west = ['SAC', 'DAL', 'POR', 'HOU', 'UTA', 'DEN', 'MEM', 'GSW', 'PHO', 'SAS', 'MIN', 'LAC', 'LAL', 'OKC', 'NOP']
    east = list(set(df['Tm'].unique()).difference(west))
    east.remove('TOT')
    teams = {"East": east, "West": west}

    for team in df['Tm'].unique():
        for conference, conference_teams in teams.items():
            if team in conference_teams:
                df['Tm'].replace({team: conference}, inplace=True)

    positions = {'Guard': ['PG', 'SG', 'PG-SG', 'SG-PF', 'SG-PG', 'SG-SF'],
                 'Forward': ['SF', 'PF', 'C', 'PF-C', 'SF-SG', 'C-SF', 'PF-SF', 'C-PF', 'SF-PF']}
    for position in df['Pos'].unique():
        for key, val in positions.items():
            if position in val:
                df['Pos'].replace({position: key}, inplace=True)

    df['3P%'].fillna(0, inplace=True)
    df['FT%'].fillna(0, inplace=True)
    df['2P%'].fillna(0, inplace=True)

    df = df[df['Year'] < 2017]
    df = df[df['Tm'] != 'TOT']
    df.dropna(axis=0, how='any', inplace=True)

    return df


def train_model():
    players = pd.read_csv("./datasets/players.csv")
    X = players.drop('All_Star', axis=1)
    y = players['All_Star']
    X.drop(columns=['Player', 'Age'], axis=1, inplace=True)
    cross_validating_groups = X['Year']
    X.drop('Year', axis=1, inplace=True)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    C = 0.01
    penalty = 'l2'
    lsvc = LinearSVC(C=C, penalty=penalty, dual=False)
    logo = LeaveOneGroupOut()
    X = np.array(X)
    y = np.array(y)
    for train_index, test_index in logo.split(X, y, groups=cross_validating_groups):
        X_cvtrain = X[train_index]
        y_cvtrain = y[train_index]
        lsvc.fit(X_cvtrain, y_cvtrain)

    return lsvc
