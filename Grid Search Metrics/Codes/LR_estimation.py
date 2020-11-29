####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
import os
import json
import argparse

from datetime import datetime
import time

import progressbar
from time import sleep

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, brier_score_loss

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

from utils import epoch_to_date, binomial_deviance
from transformations import log_transformation, standard_scale
from validation import KfoldsCV

####################################################################################################################################
####################################################################################################################################
#############################################################SETTINGS###############################################################

start_time_all = datetime.now()

# Setting arguments for running the script:
parser = argparse.ArgumentParser(description='.')
parser.add_argument("--export", default='yes', help='Argument to declare whether to export results.')
parser.add_argument("--stores", default='1424, 2056, 3859, 6044, 6256', help='Stores argument.')
parser.add_argument("--log", default='yes', help='Argument to declare the use of logarithmic transformation.')
parser.add_argument("--metric", default='roc_auc', help='Argument to declare which performance metric to use during grid search. Choose between "roc_auc", "avg_precision_score", and "brier_loss"')
args = parser.parse_args()

# Extracting inputs from arguments:
export = str.lower(args.export)
S = [int(x) for x in args.stores.split(',')]
log_transform = str.lower(args.log)
metric = str.lower(args.metric)

print('\n')
print('------------------------------------------------------------------------------------------')
print('\033[1mDEFINITIONS:\033[0m')

print('\033[1mNumber of stores:\033[0m ' + str(len(S)) + '.')
print('\033[1mStores:\033[0m ' + str(S) + '.')
print('\033[1mEstimation method:\033[0m logistic regression.')
print('\033[1mPerformance metrics:\033[0m ROC-AUC, average precision score, precision-recall-AUC, binomial-deviance, and Brier scores.')
print('------------------------------------------------------------------------------------------')
print('\n')

estimation_id = str(int(time.time()))

# Dataset info and performance metrics:
os.chdir('../Datasets')

if 'LR_grid_search_' + metric + '.json' not in os.listdir('../Datasets'):
	outcomes = {}
	
else:
	with open('LR_grid_search_' + metric + '.json') as json_file:
		outcomes = json.load(json_file)

bar = progressbar.ProgressBar(maxval=len(S), widgets=['\033[1mExecution progress:\033[0m', progressbar.Bar('=', '[', ']'), ' ',
													  progressbar.Percentage()])
bar.start()

####################################################################################################################################
####################################################################################################################################
#############################################################DATA IMPORT############################################################

for s in S:
	print('\n')
	print('------------------------------------------------------------------------------------------')
	print('\033[1mSTORE ' + str(s) + ' (' + str(S.index(s)+1) + '/' + str(len(S)) + ')\033[0m')
	print('\n')

	start_time_store = datetime.now()

	print('******************************************************************************************')
	print('\033[1mSTAGE OF DATA IMPORT\033[0m')
	print('******************************************************************************************')
	print('\n')

	print('------------------------------')
	print('\033[1mTasks:\033[0m')
	print('Importing features and labels.')
	print('------------------------------')
	print('\n')

####################################################################################################################################
# Fraud data:

	# Train data:
	os.chdir('/home/matheus_rosso/Arquivo/Features/Datasets/')

	df_train = pd.read_csv('dataset_' + str(s) + '.csv', dtype={'order_id': str, 'store_id': int})
	df_train.drop_duplicates(['order_id', 'epoch', 'order_amount'], inplace=True)

	# Date variable:
	df_train['date'] = df_train.epoch.apply(epoch_to_date)

	# Train-test split:
	df_train['train_test'] = 'test'
	df_train['train_test'].iloc[:int(df_train.shape[0]/2)] = 'train'

	df_test = df_train[df_train['train_test'] == 'test']
	df_train = df_train[df_train['train_test'] == 'train']

	print('\033[1mShape of df_train for store ' + str(s) + ':\033[0m ' + str(df_train.shape) + '.')
	print('\033[1mShape of df_test for store ' + str(s) + ':\033[0m ' + str(df_test.shape) + '.')
	print('\n')

	# Accessory variables:
	drop_vars = ['y', 'order_amount', 'store_id', 'order_id', 'status', 'epoch', 'date', 'weight', 'train_test']

	print('******************************************************************************************')
	print('\033[1mSTAGE OF DATA IMPORT: finished!\033[0m')
	print('******************************************************************************************')
	print('\n')

####################################################################################################################################
####################################################################################################################################
#############################################################DATA PRE-PROCESSING####################################################

	print('******************************************************************************************')
	print('\033[1mSTAGE OF DATA PRE-PROCESSING\033[0m')
	print('******************************************************************************************')
	print('\n')

	print('-------------------------------------------------------------------------------------------------------------------')
	print('\033[1mTasks:\033[0m')
	print('Log-transforming numerical features, standardizing numerical features, and assessing missing values.')
	print('-------------------------------------------------------------------------------------------------------------------')
	print('\n')

####################################################################################################################################
# Logarithmic transformation:

	# Variables that should not be log-transformed:
	not_log = []
	# for f in consistent_columns:
	for f in df_train.columns:
	    if ('C#' in f) | ('NA#' in f) | (f in drop_vars):
	        not_log.append(f)

	# Train data:
	print('\033[1mTrain data:\033[0m')
	if log_transform == 'yes':
	    # Assessing missing values (before logarithmic transformation):
	    if df_train.isnull().sum().sum() > 0:
	        print('\033[1mProblem - Number of overall missings detected (before logarithmic transformation):\033[0m ' +
	              str(df_train.isnull().sum().sum()) + '.')
	        print('\n')

	    log_transf = log_transformation(not_log=not_log)
	    log_transf.transform(df_train)
	    df_train = log_transf.log_transformed
	    print('\n')
	    
	    # Assessing missing values (after logarithmic transformation):
	    if df_train.isnull().sum().sum() > 0:
	        print('\033[1mProblem - Number of overall missings detected (after logarithmic transformation):\033[0m ' + 
	              str(df_train.isnull().sum().sum()) + '.')
	        print('\n')

	# Test data:
	print('\033[1mTest data:\033[0m')
	if log_transform == 'yes':
	    # Assessing missing values (before logarithmic transformation):
	    if df_test.isnull().sum().sum() > 0:
	        print('\033[1mProblem - Number of overall missings (before logarithmic transformation):\033[0m ' +
	              str(df_test.isnull().sum().sum()) + '.')
	        print('\n')

	    log_transf = log_transformation(not_log=not_log)
	    log_transf.transform(df_test)
	    df_test = log_transf.log_transformed
	    print('\n')

	    # Assessing missing values (after logarithmic transformation):
	    if df_test.isnull().sum().sum() > 0:
	        print('\033[1mProblem - Number of overall missings (after logarithmic transformation):\033[0m ' + 
	              str(df_test.isnull().sum().sum()) + '.')
	        print('\n')

####################################################################################################################################
# Standardizing numerical features:

	# Inputs that should not be standardized:
	not_stand = []
	for f in list(df_train.columns):
	    if ('C#' in f) | ('NA#' in f) | (f in drop_vars):
	        not_stand.append(f)

	# Training set:
	stand_scale = standard_scale(not_stand = not_stand)
	stand_scale.scale(train = df_train, test = df_test)
	df_train_scaled = stand_scale.train_scaled
	print('\033[1mShape of df_train_scaled:\033[0m ' + str(df_train_scaled.shape))

	# Assessing missing values (after standardizing numerical features):
	if df_train_scaled.isnull().sum().sum() > 0:
	    print('\033[1mProblem - Number of overall missings:\033[0m ' + str(df_train_scaled.isnull().sum().sum()) + '.')
	    print('\n')
	else:
	    print('\033[1mNo missing values detected (train data)!\033[0m')
	    print('\n')

	# Test set:
	df_test_scaled = stand_scale.test_scaled
	print('\033[1mShape of df_test_scaled:\033[0m ' + str(df_test_scaled.shape))

	# Assessing missing values (after standardizing numerical features):
	if df_test_scaled.isnull().sum().sum() > 0:
	    print('\033[1mProblem - Number of overall missings:\033[0m ' + str(df_test_scaled.isnull().sum().sum()) + '.')
	    print('\n')
	else:
	    print('\033[1mNo missing values detected (test data)!\033[0m')
	    print('\n')

	print('******************************************************************************************')
	print('\033[1mSTAGE OF DATA PRE-PROCESSING: finished!\033[0m')
	print('******************************************************************************************')
	print('\n')

####################################################################################################################################
####################################################################################################################################
#####################################################CV DATA MODELING###############################################################

	print('******************************************************************************************')
	print('\033[1mSTAGE OF CV DATA MODELING\033[0m')
	print('******************************************************************************************')
	print('\n')

	print('-----------------------------------------------------')
	print('\033[1mTasks:\033[0m')
	print('Performing grid search and estimating CV scores.')
	print('\033[1mEstimation method:\033[0m logistic regression.')
	print('\033[1mReference metric of performance:\033[0m ' + metric + '.')
	print('-----------------------------------------------------')
	print('\n')

	start_time = datetime.now()

	try:
		# Creating K-folds CV object:
		kfolds = KfoldsCV(method='logistic_regression', metric=metric, num_folds=3,
                  		  grid_param={'C': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1,
                                            0.25, 0.3, 0.5, 0.75, 1, 3, 10]},
                          default_param={'C': 1.0})

		# Running K-folds CV:
		kfolds.run(inputs = df_train_scaled.drop(drop_vars, axis=1), output = df_train_scaled['y'])

		# Defining best tuning hyper-parameter:
		best_param = kfolds.best_param

	except Exception as e:
		print('**********************************************************')
		print('\033[1mProblem - Error when executing CV estimation:\033[0m')
		print(e)
		print('**********************************************************')
		print('\n')

	end_time = datetime.now()

	print('\n')
	print('\033[1mRunning time (CV estimation):\033[0m ' + str(round(((end_time - start_time).seconds)/60, 2)) + ' minutes.')
	print('Start time: ' + start_time.strftime('%Y-%m-%d') + ', ' + start_time.strftime('%H:%M:%S'))
	print('End time: ' + end_time.strftime('%Y-%m-%d') + ', ' + end_time.strftime('%H:%M:%S'))
	print('\n')

	print('******************************************************************************************')
	print('\033[1mSTAGE OF CV DATA MODELING: finished!\033[0m')
	print('******************************************************************************************')
	print('\n')

####################################################################################################################################
####################################################################################################################################
###############################################TRAIN-TEST SPLIT DATA MODELING#######################################################

	print('******************************************************************************************')
	print('\033[1mSTAGE OF TRAIN-TEST SPLIT DATA MODELING\033[0m')
	print('******************************************************************************************')
	print('\n')

	print('-----------------------------------------------')
	print('\033[1mTasks:\033[0m')
	print('Estimating test scores and performance metrics.')
	print('\033[1mEstimation method:\033[0m logistic regression.')
	print('-----------------------------------------------')
	print('\n')

	start_time = datetime.now()

	try:
		# Creating estimation object:
		model = LogisticRegression(solver='liblinear',
	                               penalty = 'l1',
	                               C = best_param['C'],
	                               warm_start=True)

		# Running estimation:
		model.fit(df_train_scaled.drop(drop_vars, axis=1), df_train_scaled['y'])

		# Predicting scores:
		score_pred = [p[1] for p in model.predict_proba(df_test_scaled.drop(drop_vars, axis=1))]

		# Calculating performance metrics:
		test_roc_auc = roc_auc_score(df_test_scaled['y'], score_pred)
		test_prec_avg = average_precision_score(df_test_scaled['y'], score_pred)
		prec, rec, thres = precision_recall_curve(df_test_scaled['y'], score_pred)
		test_pr_auc = auc(rec, prec)
		test_brier = brier_score_loss(df_test_scaled['y'], score_pred)

	except Exception as e:
		test_roc_auc = np.NaN
		test_prec_avg = np.NaN
		test_pr_auc = np.NaN
		test_brier = np.NaN

		print('**********************************************************')
		print('\033[1mProblem - Error when executing train-test estimation:\033[0m')
		print(e)
		print('**********************************************************')
		print('\n')

	end_time = datetime.now()

	print('\n')
	print('\033[1mSummary (store ' + str(s) + '):\033[0m')
	print('\033[1mBest value of tuning hyper-parameter:\033[0m ' + str(best_param) + '.')
	print('\033[1mTest ROC-AUC score:\033[0m ' + str(round(test_roc_auc, 6)) + '.')
	print('\033[1mTest average precision score:\033[0m ' + str(round(test_prec_avg, 6)) + '.')
	print('\n')

	print('\033[1mRunning time (train-test estimation):\033[0m ' + str(round(((end_time - start_time).seconds)/60, 2)) +
		  ' minutes.')
	print('Start time: ' + start_time.strftime('%Y-%m-%d') + ', ' + start_time.strftime('%H:%M:%S'))
	print('End time: ' + end_time.strftime('%Y-%m-%d') + ', ' + end_time.strftime('%H:%M:%S'))
	print('\n')

	print('******************************************************************************************')
	print('\033[1mSTAGE OF TRAIN-TEST SPLIT DATA MODELING: finished!\033[0m')
	print('******************************************************************************************')
	print('\n')

####################################################################################################################################
####################################################################################################################################
######################################################ANALYSIS OF RESULTS###########################################################

	print('******************************************************************************************')
	print('\033[1mSTAGE OF ANALYSIS OF RESULTS\033[0m')
	print('******************************************************************************************')
	print('\n')

	print('-------------------------------')
	print('\033[1mTasks:\033[0m')
	print('Collecting performance metrics.')
	print('-------------------------------')
	print('\n')

	end_time_store = datetime.now()

	os.chdir('/home/matheus_rosso/Arquivo/Materiais/Codes/grid_search_metric/Datasets/')

	outcomes[s] = {
	"estimation_id": estimation_id,
	"metric": metric,
	"store_id": s,
	"n_orders_train": int(df_train_scaled.shape[0]),
	"n_orders_test": int(df_test_scaled.shape[0]),
	"n_vars": str(df_train_scaled.shape[1]),
	"first_date_train": str(str(df_train_scaled.date.min().date())),
	"last_date_train": str(df_train_scaled.date.max().date()),
	"first_date_test": str(str(df_test_scaled.date.min().date())),
	"last_date_test": str(df_test_scaled.date.max().date()),
	"avg_order_amount_train":df_train_scaled.order_amount.mean(),
	"avg_order_amount_test":df_test_scaled.order_amount.mean(),
	"method": 'logistic_regression',
	"best_param": str(best_param),
	"test_roc_auc": test_roc_auc,
	"test_prec_avg": test_prec_avg,
	"test_pr_auc": test_pr_auc,
	"test_brier_score": test_brier,
	"running_time": str(round(((end_time_store - start_time_store).seconds)/60, 2)) + ' minutes'
	}

	if export == 'yes':
		with open('LR_grid_search_' + metric + '.json', 'w') as json_file:
			json.dump(outcomes, json_file, indent=2)

	print('******************************************************************************************')
	print('\033[1mSTAGE OF ANALYSIS OF RESULTS: finished!\033[0m')
	print('******************************************************************************************')
	print('\n')

	print('\033[1mRunning time for store ' + str(s) + ':\033[0m ' +
		  str(round(((end_time_store - start_time_store).seconds)/60, 2)) + ' minutes.')
	print('Start time: ' + start_time_store.strftime('%Y-%m-%d') + ', ' + start_time_store.strftime('%H:%M:%S'))
	print('End time: ' + end_time_store.strftime('%Y-%m-%d') + ', ' + end_time_store.strftime('%H:%M:%S'))
	print('\n')

	print('\033[1mSTORE ' + str(s) + ': finished! (' + str(S.index(s)+1) + '/' + str(len(S)) + ')\033[0m')
	print('------------------------------------------------------------------------------------------')
	print('\n')

	bar.update(S.index(s)+1)
	sleep(0.1)

	# Assessing last estimation:
	with open('../last_estimation.json', 'w') as json_file:
		json.dump({'last_estimation': str(S.index(s)+1) + ' out of ' + str(len(S))}, json_file, indent=2)

# Assessing overall running time:
end_time_all = datetime.now()

print('\n')
print('------------------------------------')
print('\033[1mOverall running time:\033[0m ' + str(round(((end_time_all - start_time_all).seconds)/60, 2)) + ' minutes.')
print('Start time: ' + start_time_all.strftime('%Y-%m-%d') + ', ' + start_time_all.strftime('%H:%M:%S'))
print('End time: ' + end_time_all.strftime('%Y-%m-%d') + ', ' + end_time_all.strftime('%H:%M:%S'))
print('\n')
