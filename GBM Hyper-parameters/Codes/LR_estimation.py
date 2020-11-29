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

####################################################################################################################################
####################################################################################################################################
#############################################################SETTINGS###############################################################

start_time_all = datetime.now()

# Setting arguments for running the script:
parser = argparse.ArgumentParser(description='.')
parser.add_argument("--export", default='yes', help='Argument to declare whether to export results.')
parser.add_argument("--stores", default='1424, 2056, 3859, 6044, 6256', help='Stores argument.')
parser.add_argument("--log", default='yes', help='Argument to declare the use of logarithmic transformation.')
parser.add_argument("--regul_param", default=1.0, help='Parameter for L1 regularization.')
parser.add_argument("--grid_param", help='Values to explore for tuning hyper-parameter.')
args = parser.parse_args()

# Extracting inputs from arguments:
export = str.lower(args.export)
S = [int(x) for x in args.stores.split(',')]
log_transform = str.lower(args.log)
grid_param = [float(x) for x in args.grid_param.split(',')]

# Dictionary with hyper-parameters of logistic regression:
param_dict = {
	'C': args.regul_param
}

# Defining which hyper-parameter should be tuned:
for p in param_dict.keys():
	if param_dict[p] == 'tun_param':
		tun_param = p

# Dictionary of outputs:
outputs = {}

print('\n')
print('------------------------------------------------------------------------------------------')
print('\033[1mDEFINITIONS:\033[0m')

print('\033[1mNumber of stores:\033[0m ' + str(len(S)) + '.')
print('\033[1mStores:\033[0m ' + str(S) + '.')
print('\033[1mEstimation method:\033[0m logistic regression.')
print('\033[1mPerformance metrics:\033[0m ROC-AUC, average precision score, precision-recall-AUC, binomial-deviance, and Brier scores.')
print('\033[1mHyper-parameters:\033[0m')
print(param_dict)
print('------------------------------------------------------------------------------------------')
print('\n')

estimation_id = str(int(time.time()))

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

	start_time_test = datetime.now()

	# Logistic regression estimation:
	test_roc_auc = {}
	test_prec_avg = {}
	test_pr_auc = {}
	test_deviance = {}
	test_brier = {}

	bar_grid = progressbar.ProgressBar(maxval=len(grid_param), widgets=['\033[1mGrid estimation progress (store ' + str(s) +
																		'):\033[0m ',
	                                                             		progressbar.Bar('-', '[', ']'), ' ',
	                                                             		progressbar.Percentage()])
	bar_grid.start()

	for param in grid_param:
		try:
			# Logistic regression model:
			# Create the estimation object:
			model = LogisticRegression(solver='liblinear',
			                           penalty = 'l1',
			                           C = param,
			                           warm_start=True)

			# Training the model:
			model.fit(df_train_scaled.drop(drop_vars, axis=1),
					  df_train_scaled['y'])

			# Predicting scores and calculating performance metrics:
			score_pred = [p[1] for p in model.predict_proba(df_test_scaled.drop(drop_vars, axis=1))]

			test_roc_auc[param] = roc_auc_score(df_test_scaled['y'], score_pred)
			test_prec_avg[param] = average_precision_score(df_test_scaled['y'], score_pred)
			prec, rec, thres = precision_recall_curve(df_test_scaled['y'], score_pred)
			test_pr_auc[param] = auc(rec, prec)
			test_deviance[param] = binomial_deviance(df_test_scaled['y'], score_pred)
			test_brier[param] = brier_score_loss(df_test_scaled['y'], score_pred)

		except:
			test_roc_auc[param] = np.NaN
			test_prec_avg[param] = np.NaN
			test_pr_auc[param] = np.NaN
			test_deviance[param] = np.NaN
			test_brier[param] = np.NaN

			print('\033[1mNot able to perform train-test logistic regression estimation for store ' + str(s) + '!\033[0m')
			print('\n')

		j = grid_param.index(param)
		bar_grid.update(j+1)
		sleep(0.1)

	end_time_test = datetime.now()

	print('\n')
	print('\033[1mSummary (store ' + str(s) + '):\033[0m')
	print('\033[1mTuning hyper-parameter:\033[0m ' + tun_param + '.')
	print('\033[1mTest ROC-AUC score:\033[0m')
	print(test_roc_auc)
	print('\033[1mTest average precision score:\033[0m')
	print(test_prec_avg)
	print('\n')

	print('\033[1mRunning time:\033[0m ' + str(round(((end_time_test - start_time_test).seconds)/60, 2)) + ' minutes.')
	print('Start time: ' + start_time_test.strftime('%Y-%m-%d') + ', ' + start_time_test.strftime('%H:%M:%S'))
	print('End time: ' + end_time_test.strftime('%Y-%m-%d') + ', ' + end_time_test.strftime('%H:%M:%S'))
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

	outputs[str(s)] = {
	"estimation_id": estimation_id,
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
	"fraud_rate_train": df_train_scaled['y'].mean(),
	"fraud_rate_test": df_test_scaled['y'].mean(),
	"method": 'logistic_regression',
	"test_roc_auc": test_roc_auc,
	"test_prec_avg": test_prec_avg,
	"test_pr_auc": test_pr_auc,
	"test_deviance": test_deviance,
	"test_brier_score": test_brier,
	"running_time": str(round(((end_time_store - start_time_store).seconds)/60, 2)) + ' minutes'
	}

	# Exporting final results:
	os.chdir('/home/matheus_rosso/Arquivo/Materiais/Codes/gbm_parameters')

	if export == 'yes':
		with open('Datasets/tun_' + tun_param + '.json', 'w') as json_file:
		    json.dump(outputs, json_file, indent=2)

	print('******************************************************************************************')
	print('\033[1mSTAGE OF ANALYSIS OF RESULTS: finished!\033[0m')
	print('******************************************************************************************')
	print('\n')

	print('\033[1mRunning time for store ' + str(s) + ':\033[0m ' +
		  str(round(((end_time_store - start_time_store).seconds)/60, 2)) + ' minutes.')
	print('Start time: ' + start_time_store.strftime('%Y-%m-%d') + ', ' + start_time_store.strftime('%H:%M:%S'))
	print('End time: ' + end_time_store.strftime('%Y-%m-%d') + ', ' + end_time_store.strftime('%H:%M:%S'))
	print('\n')

	print('\033[1mSTORE ' + str(s) + ': finished! (' + str(s+1) + '/' + str(len(S)) + ')\033[0m')
	print('------------------------------------------------------------------------------------------')
	print('\n')

	bar.update(S.index(s)+1)
	sleep(0.1)

# Assessing overall running time:
end_time_all = datetime.now()

print('\n')
print('------------------------------------')
print('\033[1mOverall running time:\033[0m ' + str(round(((end_time_all - start_time_all).seconds)/60, 2)) + ' minutes.')
print('Start time: ' + start_time_all.strftime('%Y-%m-%d') + ', ' + start_time_all.strftime('%H:%M:%S'))
print('End time: ' + end_time_all.strftime('%Y-%m-%d') + ', ' + end_time_all.strftime('%H:%M:%S'))
print('\n')
