####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import numpy as np
import pandas as pd

import argparse
import os
import json

from datetime import datetime
import time

import progressbar
from time import sleep

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, brier_score_loss

####################################################################################################################################
####################################################################################################################################
#############################################################FUNCTIONS##############################################################

# Converting epoch into date:
def epoch_to_date(x):
    str_datetime = time.strftime('%d %b %Y', time.localtime(x/1000))
    dt = datetime.strptime(str_datetime, '%d %b %Y')
    return dt

# Logarithm of numerical features:
def log_transformation(x):
	"""Since numerical features are not expected to assume negative values here, and since, after a sample
	assessment, only a few negative values were identified for just a few variables, suggesting the occurrence of
	technical issues for such observations, any negative values will be truncated to zero when performing
	log-transformation."""
	if x < 0:
		new_value = 0
	else:
		new_value = x

	transf_value = np.log(new_value + 0.0001)

	return transf_value

# Function that calculates binomial deviance for a set of data points:
def binomial_deviance(y, p):
    "y is a true binary label, while p is an estimated probability for reference class."
    return np.sum(np.log(1 + np.exp(-2*y*p)))

####################################################################################################################################
####################################################################################################################################
#############################################################SETTINGS###############################################################

start_time_all = datetime.now()

# Setting arguments for running the script:
parser = argparse.ArgumentParser(description='.')
parser.add_argument("--export", default='yes', help='Argument to declare whether to export results.')
parser.add_argument("--stores", default='1424, 2056, 3859, 6044, 6256', help='Stores argument.')
parser.add_argument("--log", default='yes', help='Argument to declare the use of logarithmic transformation.')
parser.add_argument("--subsample", default=1.0, help='Subsample argument: float (subsample <= 1.0) or "tun_param".')
parser.add_argument("--max_depth", default=5, help='Number of nodes argument: integer (max_depth >= 1) or "tun_param".')
parser.add_argument("--learning_rate", default=0.1, help='Learning rate argument: float (learning_rate < 1.0) or "tun_param".')
parser.add_argument("--n_estimators", default=100, help='Number of estimators argument: integer (n_estimators > 0) or "tun_param".')
parser.add_argument("--grid_param", help='Values to explore for tuning hyper-parameter.')
args = parser.parse_args()

# Extracting inputs from arguments:
export = str.lower(args.export)
S = [int(x) for x in args.stores.split(',')]
log_transform = str.lower(args.log)
grid_param = [float(x) for x in args.grid_param.split(',')]

# Dictionary with hyper-parameters of GBM:
param_dict = {
	'subsample': args.subsample,
	'max_depth': args.max_depth,
	'learning_rate': args.learning_rate,
	'n_estimators': args.n_estimators
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
print('\033[1mEstimation method:\033[0m GBM.')
print('\033[1mPerformance metrics:\033[0m ROC-AUC, average precision score, precision-recall-AUC, binomial-deviance, and Brier scores.')
print('\033[1mHyper-parameters:\033[0m')
print(param_dict)
print('------------------------------------------------------------------------------------------')
print('\n')

bar = progressbar.ProgressBar(maxval=len(S), widgets=['\033[1mExecution progress:\033[0m', progressbar.Bar('=', '[', ']'), ' ',
													  progressbar.Percentage()])
bar.start()

####################################################################################################################################
####################################################################################################################################
#############################################################DATA IMPORT############################################################

for s in np.arange(0, len(S)):
	print('\n')
	print('------------------------------------------------------------------------------------------')
	print('\033[1mSTORE ' + str(S[s]) + ' (' + str(s+1) + '/' + str(len(S)) + ')\033[0m')
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

	os.chdir('/home/matheus_rosso/Arquivo/Features/Datasets')

	# Importing training dataset:
	df = pd.read_csv('dataset_' + str(S[s]) + '.csv', dtype={'order_id':str})

	# Dropping duplicates that follow from the original data:
	df.drop_duplicates(subset=['order_id', 'epoch', 'order_amount'], inplace=True)

	print('\033[1mShape of df:\033[0m ' + str(df.shape) + '.')
	print('\n')

	df['date'] = df['epoch'].apply(epoch_to_date)

	# Basic dataset information:
	df_info_dict = {
	"store_id":S[s],
	"shape":df.shape,
	"time_period":(str(df['date'].min().date()), str(df['date'].max().date())),
	"fraud_rate":df['y'].mean(),
	"avg_order_amount":df['order_amount'].mean()
	}

	print('\033[1mBasic dataset information:\033[0m')
	print(df_info_dict)
	print('\n')

	# Train-test split:
	df['train_test'] = 'test'
	df['train_test'].iloc[:int(df.shape[0]/2)] = 'train'
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
	print('Log-transforming numerical features, assessing missing values and standardizing features.')
	print('-------------------------------------------------------------------------------------------------------------------')
	print('\n')

	#################################################################################################################################
	# Logarithm of numerical features:
	# Assessing missing values (before logarithmic transformation):
	if df.isnull().sum().sum() > 0:
		print('\033[1mNumber of overall missings (before logarithmic transformation):\033[0m ' + str(df.isnull().sum().sum()) + '.')
		print('\n')

	if log_transform == 'yes':
		new_col = []
		log_vars = []

		for f in list(df.columns):
		    if ('C#' in f) | ('NA#' in f) | (f in drop_vars):
		        new_col.append(f)
		    else:
		        new_col.append('L#' + f)
		        log_vars.append('L#' + f)
		        df[f] = df[f].apply(log_transformation)

		print('\033[1mNumber of numerical features log-transformed:\033[0m ' + str(len(log_vars)) + ' out of ' +
			  str(df.shape[1] - len(drop_vars)) + '.')

		df.columns = new_col

		# Assessing missing values (after logarithmic transformation):
		if df.isnull().sum().sum() > 0:
			print('\033[1mNumber of overall missings (after logarithmic transformation):\033[0m ' + str(df.isnull().sum().sum()) + '.')
			print('\n')

	#################################################################################################################################
	# Standardizing numerical features:
	# Inputs that should not be standardized:
	not_stand = []
	for f in list(df.columns):
		if ('C#' in f) | ('NA#' in f):
			not_stand.append(f)

	# Training set:
	scaler = StandardScaler()
	scaler.fit(df[df['train_test']=='train'].drop(drop_vars, axis=1).drop(not_stand, axis=1))
	num_var_stand_train = scaler.transform(df[df['train_test']=='train'].drop(drop_vars, axis=1).drop(not_stand, axis=1))
	num_var_stand_train = pd.DataFrame(data=num_var_stand_train,
									   columns=df[df['train_test']=='train'].drop(drop_vars, axis=1).drop(not_stand, axis=1).columns)
	num_var_stand_train.index = df[df['train_test']=='train'].index
	print('\033[1mShape of num_var_stand (training set):\033[0m ' + str(num_var_stand_train.shape))

	# Test set:
	num_var_stand_test = scaler.transform(df[df['train_test']=='test'].drop(drop_vars, axis=1).drop(not_stand, axis=1))
	num_var_stand_test = pd.DataFrame(data=num_var_stand_test,
									   columns=df[df['train_test']=='test'].drop(drop_vars, axis=1).drop(not_stand, axis=1).columns)
	num_var_stand_test.index = df[df['train_test']=='test'].index
	print('\033[1mShape of num_var_stand (test set):\033[0m ' + str(num_var_stand_test.shape))

	num_var_stand = pd.concat([num_var_stand_train, num_var_stand_test], axis=0, sort=False)

	df_num_var_stand = pd.concat([df[drop_vars], df[not_stand], num_var_stand], axis=1)
	print('\033[1mShape of df_num_var_stand:\033[0m ' + str(df_num_var_stand.shape))
	print('\n')

	# Assessing missing values (after standardizing numerical features):
	if df_num_var_stand.isnull().sum().sum() > 0:
		print('\033[1mNumber of overall missings:\033[0m ' + str(df_num_var_stand.isnull().sum().sum()) + '.')
		print('\n')
	else:
		print('\033[1mNo missing values detected!\033[0m')
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
	print('\033[1mEstimation method:\033[0m GBM.')
	print('-----------------------------------------------')
	print('\n')

	start_time_test = datetime.now()

	# GBM estimation:
	test_roc_auc = {}
	test_prec_avg = {}
	test_pr_auc = {}
	test_deviance = {}
	test_brier = {}

	bar_grid = progressbar.ProgressBar(maxval=len(grid_param), widgets=['\033[1mGrid estimation progress (store ' + str(S[s]) +
																		'):\033[0m ',
	                                                             		progressbar.Bar('-', '[', ']'), ' ',
	                                                             		progressbar.Percentage()])
	bar_grid.start()

	for param in grid_param:
		try:
		    # Gradient boosting model:
		    # Create the estimation object:
		    if tun_param == 'subsample':
		    	model = GradientBoostingClassifier(subsample = param, max_depth = int(param_dict['max_depth']),
												   learning_rate = float(param_dict['learning_rate']),
												   n_estimators = int(param_dict['n_estimators']), warm_start=True)
		    elif tun_param == 'max_depth':
		    	model = GradientBoostingClassifier(subsample = float(param_dict['subsample']), max_depth = int(param),
												   learning_rate = float(param_dict['learning_rate']),
												   n_estimators = int(param_dict['n_estimators']), warm_start=True)
		    elif tun_param == 'learning_rate':
		    	model = GradientBoostingClassifier(subsample = float(param_dict['subsample']),
		    									   max_depth = int(param_dict['max_depth']), learning_rate = param,
		    									   n_estimators = int(param_dict['n_estimators']), warm_start=True)
		    else:
		    	model = GradientBoostingClassifier(subsample = float(param_dict['subsample']),
		    									   max_depth = int(param_dict['max_depth']),
		    									   learning_rate = float(param_dict['learning_rate']), n_estimators = int(param),
		    									   warm_start=True)

		    # Training the model:
		    model.fit(df_num_var_stand[df_num_var_stand['train_test'] == 'train'].drop(drop_vars, axis=1),
		    		  df_num_var_stand[df_num_var_stand['train_test'] == 'train']['y'])

		    # Predicting scores and calculating performance metrics:
		    score_pred = [p[1] for p in model.predict_proba(df_num_var_stand[df_num_var_stand['train_test'] == 'test'].drop(drop_vars,
		                                                                                                                        axis=1))]
		    
		    test_roc_auc[param] = roc_auc_score(df_num_var_stand[df_num_var_stand['train_test'] == 'test']['y'], score_pred)
		    test_prec_avg[param] = average_precision_score(df_num_var_stand[df_num_var_stand['train_test'] == 'test']['y'], score_pred)
		    prec, rec, thres = precision_recall_curve(df_num_var_stand[df_num_var_stand['train_test'] == 'test']['y'], score_pred)
		    test_pr_auc[param] = auc(rec, prec)
		    test_deviance[param] = binomial_deviance(df_num_var_stand[df_num_var_stand['train_test'] == 'test']['y'], score_pred)
		    test_brier[param] = brier_score_loss(df_num_var_stand[df_num_var_stand['train_test'] == 'test']['y'], score_pred)

		except:
			test_roc_auc[param] = np.NaN
			test_prec_avg[param] = np.NaN
			test_pr_auc[param] = np.NaN
			test_deviance[param] = np.NaN
			test_brier[param] = np.NaN

			print('\033[1mNot able to perform train-test GBM estimation for store ' + str(S[s]) + '!\033[0m')
			print('\n')

		j = grid_param.index(param)
		bar_grid.update(j+1)
		sleep(0.1)

	end_time_test = datetime.now()

	print('\n')
	print('\033[1mSummary (store ' + str(S[s]) + '):\033[0m')
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

	outputs[str(S[s])] = {
	"store_id": S[s],
	"n_orders": int(df.shape[0]),
	"n_vars": str(df.shape[1]),
	"first_date": str(str(df['date'].min().date())),
	"last_date": str(df['date'].max().date()),
	"avg_order_amount":df['order_amount'].mean(),
	"fraud_rate": df['y'].mean(),
	"method": 'GBM',
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

	print('\033[1mRunning time for store ' + str(S[s]) + ':\033[0m ' +
		  str(round(((end_time_store - start_time_store).seconds)/60, 2)) + ' minutes.')
	print('Start time: ' + start_time_store.strftime('%Y-%m-%d') + ', ' + start_time_store.strftime('%H:%M:%S'))
	print('End time: ' + end_time_store.strftime('%Y-%m-%d') + ', ' + end_time_store.strftime('%H:%M:%S'))
	print('\n')

	print('\033[1mSTORE ' + str(S[s]) + ': finished! (' + str(s+1) + '/' + str(len(S)) + ')\033[0m')
	print('------------------------------------------------------------------------------------------')
	print('\n')

	bar.update(s+1)
	sleep(0.1)

# Assessing overall running time:
end_time_all = datetime.now()

print('\n')
print('------------------------------------')
print('\033[1mOverall running time:\033[0m ' + str(round(((end_time_all - start_time_all).seconds)/60, 2)) + ' minutes.')
print('Start time: ' + start_time_all.strftime('%Y-%m-%d') + ', ' + start_time_all.strftime('%H:%M:%S'))
print('End time: ' + end_time_all.strftime('%Y-%m-%d') + ', ' + end_time_all.strftime('%H:%M:%S'))
print('\n')
