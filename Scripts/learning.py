
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

models = [{'estimator': LogisticRegression(), 'n_jobs': -1, 'param_grid': {'C': [0.01, 0.1, 1, 10, 100],
									'dual': [True, False],
									'penalty': ['l1', 'l2'],
									'tol': [1e-3, 1e-4],
									'max_iter': [100, 500, 1000]}},
		  {'estimator': RandomForestClassifier(), 'param_grid': {'n_estimators': [100, 500, 1000], 
									'max_features': ['sqrt', None], 
									'max_depth': [3, 4, None], 
									'min_samples_split': [2, 3, 10], 
									'criterion': ['gini', 'entropy'],
									'n_jobs': [-1]}},
		  {'estimator': SVC(), 'n_jobs': -1, 'param_grid': {'C': [0.01, 0.1, 1, 10, 100], 
									'probability': [True],
									'kernel':  ['linear', 'poly', 'rbf', 'sigmoid'],
									'degree': [2, 3, 4],
									'tol': [1e-3, 1e-4]}}]

for model in models:
	model['error_score'] = 0.0

def numTimesRoadWon(df):
	return sum(df['won'] == 0)

def homeWonTrainIndices(df, train_size):
	indices = df[df['won'] == 1].index
	return np.random.choice(indices, train_size, replace=False)

def trainIndices(home_train, road_train):
	return np.append(home_train, road_train)

def testIndices(df, train_indices):
	return np.setdiff1d(df.index, train_indices)

def scoreProbabilities(df, probabilities):
	scoring = list()
	# print(clf.classes_) # verify order is [class 0, class 1]
	for truth, prob in zip(df['won'], probabilities):
		prob_road_wins = prob[0]
		prob_home_wins = prob[1]
		if prob_home_wins > 0.5:
			if truth == 1.0:
				scoring.append( (prob_home_wins, True) )
			else:
				scoring.append( (prob_home_wins, False) )
		if prob_road_wins > 0.5:
			if truth == 0.0:
				scoring.append( (prob_road_wins, True) )
			else:
				scoring.append( (prob_road_wins, False) )
	return scoring

def summarizeProbabilityScoring(scoring, step_size=0.05):
	results = dict()
	lower_bound = 0.5
	while lower_bound < 1.0:
		results[lower_bound] = {'correct': 0, 'samples': 0}
		for prob, correct in scoring:
			if prob >= lower_bound:
				if correct:
					results[lower_bound]['correct'] += 1
				results[lower_bound]['samples'] += 1
		lower_bound += step_size
	return results

def printBins(bins):
	for rng, results in sorted(bins.items(), key=lambda x: x[0], reverse=True):
		print('>= {:.2f}: {} / {} ({:.1f}%)'.format(rng,  
													 results['correct'], 
													 results['samples'], 
													 results['correct'] / results['samples'] * 100 \
													 if results['samples'] else 0.0))


def main():
	df_train = pd.read_csv('train.csv') # games from 2000-2014
	df_test = pd.read_csv('test.csv') # games from 2014-2016

	# home team wins more frequently
	# undersample these cases to equal number of cases where road team won
	num_times_road_won = numTimesRoadWon(df_train)
	# randomly sample that number from the instances where home team won
	home_train_indices = homeWonTrainIndices(df_train, num_times_road_won)
	road_train_indices = df_train[df_train['won'] == 0].index
	# combine home and road indices
	train_indices = trainIndices(home_train_indices, road_train_indices)

	for model in models:
		clf = GridSearchCV(**model)
		clf.fit(X=df_train[df_train.columns.difference(['gameId', 'won'])].loc[train_indices], y=df_train['won'].loc[train_indices])

		print('\n\n\n\nModel:', model['estimator'].__class__.__name__)
		print('\n\nTRAINING\n')
		print('Cross Validation Best Estimator')
		print(clf.best_estimator_)
		print('Cross Validation Best Params')
		print(clf.best_params_)
		print('Cross Validation Best Accuracy Score')
		print(clf.best_score_)

		# probabilities are [ [road wins prob, home wins prob] ... ]
		probabilities = clf.predict_proba(X=df_test[df_test.columns.difference(['gameId', 'won'])])
		# regular prediction
		pred = clf.predict(X=df_test[df_test.columns.difference(['gameId', 'won'])])
		print(pred)

		score_probabilities = scoreProbabilities(df_test, probabilities)
		probability_bins = summarizeProbabilityScoring(score_probabilities, step_size=0.05)
		class_report = classification_report(y_true=df_test['won'], y_pred=pred)
		conf_matrix = confusion_matrix(y_true=df_test['won'], y_pred=pred)
		accuracy = accuracy_score(y_true=df_test['won'], y_pred=pred)

		print('\n\nTESTING\n')
		print('Binned Probability Scores')
		printBins(probability_bins)
		print('Classification Report')
		print(class_report)
		print('Normalized Confusion Matrix')
		print(conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis])
		print('Accuracy')
		print(accuracy)

main()

