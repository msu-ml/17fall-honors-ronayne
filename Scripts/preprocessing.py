
import requests
import json
import pandas as pd
import numpy as np

# 2000-2001 season thru 2016-2017 season
url = 'http://www.nhl.com/stats/rest/team?isAggregate=false&reportType=basic&isGame=true&reportName=teamsummary&sort=[{%22property%22:%22points%22,%22direction%22:%22DESC%22},{%22property%22:%22wins%22,%22direction%22:%22DESC%22}]&cayenneExp=gameDate%3E=%222000-10-04%22%20and%20gameDate%3C=%222017-09-04%22%20and%20gameTypeId=2'

cumulative_sum_cols = ['faceoffsWon',
					   'faceoffsLost',
					   'gamesPlayed',
					   'goalsFor',
					   'goalsAgainst',
					   'points',
					   'ppGoalsFor',
					   'ppOpportunities',
					   'pkSaves',
					   'shNumTimes',
					   'shotsFor',
					   'shotsAgainst']

keep_cols = ['faceoffWinPctg',
			 'goalsFor',
			 'goalsAgainst',
			 'points',
			 'ppGoalPctg',
			 'pkSavePctg',
			 'shotsFor',
			 'shotsAgainst',
			 'shNumTimes',
			 'gameId',
			 'won',
			 'ties',
			 'gameLocationCode',
			 'gamesPlayed',
			 'gameDate',
			 'daysBetweenGames',
			 'teamAbbrev',
			 'winStreak',
			 'loseStreak']

def get_api_data(url):
	# get data from NHL.com/stats
	response = requests.get(url)
	content = json.loads(response.content)
	data = content['data']
	df = pd.DataFrame(data)
	df.to_csv('raw_data.csv', index=False)
	return df

def get_csv_data(filename):
	# load data from csv instead of from API every time
	return pd.read_csv(filename)

def seasonId(df):
	return df['gameId'].astype(str).str[:4]

def previousMatchups(df):
	df.sort_values(['teamAbbrev', 'opponentTeamAbbrev', 'gameDate'], inplace=True)
	df['prevMatchup'] = df.groupby([df['teamAbbrev'], df['opponentTeamAbbrev'], seasonId(df)])['won'].shift(1)

def removeLockoutSeasons(df): # return new df
	df = df[~seasonId(df).isin(['2004', '2012'])]
	return df

def parseDate(df): # inplace
	df['gameDate'] = pd.to_datetime(df['gameDate'])

def renameWinsCol(df): # inplace
	# we are going to convert the wins column to a cumsum of wins
	# need another column indicating whether team won that specific game
	df.rename(columns={'wins': 'won'}, inplace=True)

def pkSaves(df): # inplace
	# calculate successful penalty kills
	df['pkSaves'] = df['penaltyKillPctg'] * df['shNumTimes']
	df['pkSaves'] = df['pkSaves'].astype(int)

def consecutiveWins(x):
	# https://stackoverflow.com/a/35428677/7636711
	x['winStreak'] = x.groupby( (x['won'] == 0).cumsum()).cumcount() + \
					 ( (x['won'] == 0).cumsum() == 0).astype(int)
	return x

def consecutiveLosses(x):
	# https://stackoverflow.com/a/35428677/7636711
	x['loseStreak'] = x.groupby( (x['won'] != 0).cumsum()).cumcount() + \
					  ( (x['won'] != 0).cumsum() == 0).astype(int)
	return x

def streakCalc(df): # return new df
	# https://stackoverflow.com/a/35428677/7636711
	df.sort_values(['teamAbbrev', 'gameId'], inplace=True)
	df = df.groupby([df['teamAbbrev'], seasonId(df)], sort=False).apply(consecutiveWins)
	df = df.groupby([df['teamAbbrev'], seasonId(df)], sort=False).apply(consecutiveLosses)
	return df

def cumSum(df, cols): # inplace
	# sum team stats for every game leading up to each game, by season
	df.sort_values(['teamAbbrev', 'gameId'], inplace=True)
	for col in cols:
		# season ID is identified by the first four numbers of gameId
		df[col] = df[col].groupby([df['teamAbbrev'], seasonId(df)]).cumsum()

def pctgCalculations(df): # inplace
	# calculate percentages after cumsum
	df['faceoffWinPctg'] = df['faceoffsWon'] / (df['faceoffsWon'] + df['faceoffsLost'])
	df['ppGoalPctg'] = df['ppGoalsFor'] / df['ppOpportunities']
	df['pkSavePctg'] = df['pkSaves'] / df['shNumTimes']

def shiftCols(df): # inplace
	# want data UP TO a certain game
	df.sort_values(['teamAbbrev', 'gameId'], inplace=True)
	# don't shift game metadata
	for col in df.columns.difference(['gameId', 'won', 'gameLocationCode', 'gameDate', 'ties', 'teamAbbrev']):
		df[col] = df[col].groupby([df['teamAbbrev'], seasonId(df)]).shift(1) # shift down

def daysBetweenGames(df): # inplace
	# biggest differences will be after Olympic and All-Star breaks in February
	df.sort_values(['teamAbbrev', 'gameDate'], inplace=True)
	df['daysBetweenGames'] = df.groupby([df['teamAbbrev'], seasonId(df)])['gameDate'].diff() \
							 / np.timedelta64(1, 'D')
	df['daysBetweenGames'] = df['daysBetweenGames'].fillna(0)

def dropCols(df, cols): # inplace
	df.drop(cols, axis=1, inplace=True)

def keepCols(df, cols): # inplace
	df.drop(df.columns.difference(cols), axis=1, inplace=True)

def gameIdIndex(df): # inplace
	# set index to gameId
	df.set_index('gameId', inplace=True)
	df.sort_index(inplace=True)

def filterGamesPlayed(df, min_games_played): # return new df
	gameIds = df[df['gamesPlayed'] >= min_games_played].index.unique()
	df = df.loc[gameIds].copy()
	df.drop(['gamesPlayed'], axis=1, inplace=True)
	return df

def filterTies(df): # return new df
	# don't want games that ended in a tie
	# as of 2005, NHL games can no longer end in ties
	df = df[df['ties'] != 1].copy()
	df.drop(['ties'], axis=1, inplace=True)
	return df

def separateHomeRoad(df, location='H'): # return new df
	location_df = df[df['gameLocationCode'] == location].copy()
	location_df.drop(['gameLocationCode'], axis=1, inplace=True)
	return location_df

def combineHomeRoad(home, road): # return new df
	df = home[home.columns.difference(['gameDate', 'teamAbbrev'])] \
		 - road[home.columns.difference(['gameDate', 'teamAbbrev'])]
	df['won'] = home['won']
	return df

def split_df(df, first_season=2000, last_season=2016): # return new df
	lower_bound = df.index.astype(str).str[:4].astype(int) >= first_season
	upper_bound = df.index.astype(str).str[:4].astype(int) <= last_season
	return df[lower_bound & upper_bound].copy()

def meanNormalize(df): # return new df
	data = df[df.columns.difference(['won'])] - df[df.columns.difference(['won'])].mean()
	data['won'] = df['won']
	return data

def fixPrevMatchup(df): # inplace
	df['prevMatchup'] = df['prevMatchup'].replace(0.0, -1).fillna(0).astype(int)

def main():
	#df = get_api_data(url)
	df = get_csv_data('raw_data.csv')
	# parse the gameDate string into a datetime object
	parseDate(df)
	# remove shortened seasons https://en.wikipedia.org/wiki/NHL_lockout
	df = removeLockoutSeasons(df)
	# rename the wins columns to "won" because this will not be cumsummed
	renameWinsCol(df)
	# find who won the game the last time the two teams met
	#previousMatchups(df) # TODO: fully implement
	# calculate the penalty kills saves feature
	pkSaves(df)
	# calculate the number of wins in a row per team
	df = streakCalc(df)
	# calculate the cumulative sum
	# sum of all stats up to and including each row, by team, by season
	cumSum(df, cumulative_sum_cols)
	# calculate features that are based on other features
	pctgCalculations(df)
	# shift some row data up one row
	# allows each row to represent stats up to, but NOT including, that row
	shiftCols(df)
	# get rid of columns we don't need anymore, and keep the ones we do
	keepCols(df, keep_cols)
	# calculate the number of days between the each team's last game
	daysBetweenGames(df)
	# get rid of games that ended in a tie
	# this wil get rid of both rows for each game because both have ties == 1
	df = filterTies(df)
	# set index of df to the gameId
	gameIdIndex(df)
	# get rid of all games where at least one of the teams hadn't played 41 games
	# gets rid of both rows for each game meeting this criteria
	df = filterGamesPlayed(df, 41)
	# separate the rows into home and road team data
	home = separateHomeRoad(df, location='H')
	road = separateHomeRoad(df, location='R')
	# join the home and road team data into one row
	# data is home team stats minus road team stats
	df = combineHomeRoad(home, road)
	# center the data around the mean
	df = meanNormalize(df)
	# train on all games from 1994-2014 seasons
	df_train = split_df(df, 2000, 2014)
	df_train.to_csv('train.csv')
	# test on all games from 2015-2016 seasons
	df_test = split_df(df, 2015, 2016)
	df_test.to_csv('test.csv')

main()

