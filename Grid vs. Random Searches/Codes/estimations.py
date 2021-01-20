####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
import json
import os
import argparse

from datetime import datetime
import time

import progressbar
from time import sleep

from scipy.stats import uniform, norm, randint

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, brier_score_loss

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

import utils
from utils import epoch_to_date, text_clean, is_velocity, get_cat

from transformations import log_transformation, standard_scale, recreate_missings, impute_missing
from transformations import one_hot_encoding

import validation
from validation import Kfolds_fit

####################################################################################################################################
####################################################################################################################################
#############################################################SETTINGS###############################################################

start_time_all = datetime.now()

# Setting arguments for running the script:
parser = argparse.ArgumentParser(description='.')
parser.add_argument("--export", default=False, help='Argument to declare whether to export results. For exporting all outputs, choose "all". For exporting only performance metrics, choose "metrics". For no exports, choose "no".')
parser.add_argument("--stores", default='1098', help='Stores argument.')
parser.add_argument("--log", default=True, help='Argument to declare the use of logarithmic transformation over numerical data.')
parser.add_argument("--stand", default=True, help='Argument to declare whether to standardize numerical data.')
parser.add_argument("--method", default='logistic_regression', help='Argument to declare which estimation method to use (choose between "logistic_regression" and "GBM").')
parser.add_argument("--random_search", default=False, help='Choose whether to perform random search (True) or grid search (False).')
args = parser.parse_args()

# Extracting inputs from arguments:
export = str.lower(str(args.export)) == 'true'
S = [int(x) for x in args.stores.split(',')]
log_transform = str.lower(str(args.log)) == 'true'
standardize = str.lower(str(args.stand)) == 'true'
method = args.method
random_search = str.lower(str(args.random_search)) == 'true'

# Define the number of samples to implement random search:
if method == 'logistic_regression':
    n_samples = 10

elif method == 'GBM':
    n_samples = 20

print('\n')
print('----------------------------------------------------------------------------------------------')
print('\033[1mDEFINITIONS:\033[0m')

print('\033[1mNumber of stores:\033[0m ' + str(len(S)) + '.')
print('\033[1mStores:\033[0m ' + str(S) + '.')
print('\033[1mEstimation method:\033[0m logistic regression.')
print('\033[1mPerformance metrics:\033[0m ROC-AUC, average precision score, precision-recall-AUC, and Brier scores.')
print('----------------------------------------------------------------------------------------------')
print('\n')

# Dictionary with information on model structure and performance:
os.chdir('/home/matheus_rosso/Arquivo/Materiais/Codes/grid_random_searches/')

if 'model_assessment.json' not in os.listdir('Datasets'):
    model_assessment = {}

else:
    with open('Datasets/model_assessment.json') as json_file:
        model_assessment = json.load(json_file)

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

	estimation_id = str(int(time.time()))
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

	# Train data:
	os.chdir('/home/matheus_rosso/Arquivo/Features/Datasets/')

	df_train = pd.read_csv('new_additional_datasets/dataset_' + str(s) + '.csv',
	                       dtype={'order_id': str, 'store_id': int})
	df_train.drop_duplicates(['order_id', 'epoch', 'order_amount'], inplace=True)
	df_train['date'] = df_train.epoch.apply(epoch_to_date)

	# Dropping original categorical features:
	cat_vars = get_cat(df_train)
	c_vars = [c for c in list(df_train.columns) if 'C#' in c]
	na_vars = ['NA#' + c for c in cat_vars if 'NA#' + c in list(df_train.columns)]

	df_train = df_train.drop(c_vars, axis=1).drop(na_vars, axis=1)

	# Splitting data into train and test:
	df_test = df_train[(df_train.date > datetime.strptime('2020-03-30', '%Y-%m-%d'))]
	df_train = df_train[(df_train.date <= datetime.strptime('2020-03-30', '%Y-%m-%d'))]

	print('\033[1mShape of df_train for store ' + str(s) + ':\033[0m ' + str(df_train.shape) + '.')
	print('\033[1mShape of df_test for store ' + str(s) + ':\033[0m ' + str(df_test.shape) + '.')
	print('\n')

	# Accessory variables:
	drop_vars = ['y', 'order_amount', 'store_id', 'order_id', 'status', 'epoch', 'date', 'weight']

	# Assessing missing values:
	num_miss_train = df_train.isnull().sum().sum()
	num_miss_test = df_test.isnull().sum().sum()

	if num_miss_train > 0:
	    print('\033[1mProblem - Number of overall missings detected (training data):\033[0m ' +
	          str(df_train.isnull().sum().sum()) + '.')
	    print('\n')

	if num_miss_test > 0:
	    print('\033[1mProblem - Number of overall missings detected (test data):\033[0m ' +
	          str(df_test.isnull().sum().sum()) + '.')
	    print('\n')

####################################################################################################################################
# Categorical datasets:

	categorical_train = pd.read_csv('new_additional_datasets/categorical_features/dataset_' + str(s) + '.csv',
                      				dtype={'order_id': str, 'store_id': int})
	categorical_train.drop_duplicates(['order_id', 'epoch', 'order_amount'], inplace=True)

	categorical_train['date'] = categorical_train.epoch.apply(epoch_to_date)

	# Splitting data into train and test:
	categorical_test = categorical_train[(categorical_train.date > datetime.strptime('2020-03-30', '%Y-%m-%d'))]
	categorical_train = categorical_train[(categorical_train.date <= datetime.strptime('2020-03-30', '%Y-%m-%d'))]

	print('\033[1mShape of categorical_train (training data):\033[0m ' + str(categorical_train.shape) + '.')
	print('\033[1mNumber of orders (training data):\033[0m ' + str(categorical_train.order_id.nunique()) + '.')
	print('\n')

	print('\033[1mShape of categorical_test (test data):\033[0m ' + str(categorical_test.shape) + '.')
	print('\033[1mNumber of orders (test data):\033[0m ' + str(categorical_test.order_id.nunique()) + '.')
	print('\n')

	# Treating missing values:
	print('\033[1mAssessing missing values in categorical data (training data):\033[0m')
	print(categorical_train.drop(drop_vars, axis=1).isnull().sum().sort_values(ascending=False))

	print('\033[1mAssessing missing values in categorical data (test data):\033[0m')
	print(categorical_test.drop(drop_vars, axis=1).isnull().sum().sort_values(ascending=False))

	# Loop over categorical features:
	for f in categorical_train.drop(drop_vars, axis=1).columns:
	    # Training data
	    categorical_train[f] = categorical_train[f].apply(lambda x: 'NA_VALUE' if pd.isna(x) else x)
	    
	    # Test data:
	    categorical_test[f] = categorical_test[f].apply(lambda x: 'NA_VALUE' if pd.isna(x) else x)

	# Assessing missing values:
	if categorical_train.isnull().sum().sum() > 0:
	    print('\033[1mProblem - Number of overall missings detected (training data):\033[0m ' +
	          str(categorical_train.isnull().sum().sum()) + '.')
	    print('\n')

	if categorical_test.isnull().sum().sum() > 0:
	    print('\033[1mProblem - Number of overall missings detected (test data):\033[0m ' +
	          str(categorical_test.isnull().sum().sum()) + '.')
	    print('\n')

	na_vars = [c for c in categorical_train.drop(drop_vars, axis=1) if 'NA#' in c]

	# Treating text data:
	# Loop over categorical features:
	for f in categorical_train.drop(drop_vars, axis=1).drop(na_vars, axis=1).columns:
	    # Training data:
	    categorical_train[f] = categorical_train[f].apply(lambda x: text_clean(str(x)))
	    
	    # Test data:
	    categorical_test[f] = categorical_test[f].apply(lambda x: text_clean(str(x)))

####################################################################################################################################
# Merging all features:

	# Training data:
	df_train = df_train.merge(categorical_train[[f for f in categorical_train.columns if (f not in drop_vars) |
	                                             (f == 'order_id')]],
	                          on='order_id', how='left')

	print('\033[1mShape of df_train for store ' + str(s) + ':\033[0m ' + str(df_train.shape) + '.')
	print('\n')

	# Test data:
	df_test = df_test.merge(categorical_test[[f for f in categorical_test.columns if (f not in drop_vars) |
	                                          (f == 'order_id')]],
	                        on='order_id', how='left')

	print('\033[1mShape of df_test for store ' + str(s) + ':\033[0m ' + str(df_test.shape) + '.')
	print('\n')

	# Assessing missing values (training data):
	if df_train.isnull().sum().sum() != num_miss_train:
	    print('\033[1mInconsistent number of overall missings values (training data)!\033[0m')
	    print('\n')

	# Assessing missing values (test data):
	if df_test.isnull().sum().sum() != num_miss_test:
	    print('\033[1mInconsistent number of overall missings values (test data)!\033[0m')
	    print('\n')

####################################################################################################################################
# Classifying features:

	# Categorical features:
	cat_vars = list(categorical_train.drop(drop_vars, axis=1).columns)

	# Dummy variables indicating missing value status:
	missing_vars = [c for c in list(df_train.drop(drop_vars, axis=1).columns) if ('NA#' in c)]

	# Dropping features with no variance:
	no_variance = [c for c in df_train.drop(drop_vars, axis=1).drop(cat_vars,
	                                                                axis=1).drop(missing_vars,
	                                                                             axis=1) if df_train[c].var()==0]

	if len(no_variance) > 0:
	    df_train.drop(no_variance, axis=1, inplace=True)
	    df_test.drop(no_variance, axis=1, inplace=True)

	# Numerical features:
	cont_vars = [c for c in  list(df_train.drop(drop_vars, axis=1).columns) if is_velocity(c)]

	# Binary features:
	binary_vars = [c for c in list(df_train.drop([c for c in df_train.columns if (c in drop_vars) |
	                                             (c in cat_vars) | (c in missing_vars) | (c in cont_vars)],
	                                             axis=1).columns) if set(df_train[c].unique()) == set([0,1])]

	# Updating the list of numerical features:
	for c in list(df_train.drop(drop_vars, axis=1).columns):
	    if (c not in cat_vars) & (c not in missing_vars) & (c not in cont_vars) & (c not in binary_vars):
	        cont_vars.append(c)

	# Dataframe presenting the frequency of features by class:
	feats_assess = pd.DataFrame(data={
	    'class': ['cat_vars', 'missing_vars', 'binary_vars', 'cont_vars', 'drop_vars'],
	    'frequency': [len(cat_vars), len(missing_vars), len(binary_vars), len(cont_vars), len(drop_vars)]
	})
	print(feats_assess.sort_values('frequency', ascending=False))
	print('\n')

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
	print('Assessing missing values, log-transforming and standardizing numerical features, treating missing values, and transforming categorical features.')
	print('-------------------------------------------------------------------------------------------------------------------')
	print('\n')

####################################################################################################################################
# Assessing missing values:

	# Recreating missing values:
	missing_vars = [f for f in df_train.columns if 'NA#' in f]

	# Loop over variables with missing values:
	for f in [c for c in missing_vars if c.replace('NA#', '') not in cat_vars]:
	    if f.replace('NA#', '') in df_train.columns:
	        # Training data:
	        df_train[f.replace('NA#', '')] = recreate_missings(df_train[f.replace('NA#', '')], df_train[f])
	        
	        # Test data:
	        df_test[f.replace('NA#', '')] = recreate_missings(df_test[f.replace('NA#', '')], df_test[f])
	    else:
	        df_train.drop([f], axis=1, inplace=True)
	        
	        df_test.drop([f], axis=1, inplace=True)

	# Dropping all variables with missing value status:
	df_train.drop([f for f in df_train.columns if 'NA#' in f], axis=1, inplace=True)

	df_test.drop([f for f in df_test.columns if 'NA#' in f], axis=1, inplace=True)

	# Describing the frequency of missing values:
	# Dataframe with the number of missings by feature (training data):
	missings_dict = df_train.isnull().sum().sort_values(ascending=False).to_dict()

	missings_assess_train = pd.DataFrame(data={
	    'feature': list(missings_dict.keys()),
	    'missings': list(missings_dict.values())
	})

	print('\033[1mNumber of features with missings:\033[0m {}'.format(sum(missings_assess_train.missings > 0)) +
	      ' out of {} features'.format(len(missings_assess_train)) +
	      ' ({}%).'.format(round((sum(missings_assess_train.missings > 0)/len(missings_assess_train))*100, 2)))
	print('\033[1mAverage number of missings:\033[0m {}'.format(int(missings_assess_train.missings.mean())) +
	      ' out of {} observations'.format(len(df_train)) +
	      ' ({}%).'.format(round((int(missings_assess_train.missings.mean())/len(df_train))*100,2)))
	print('\n')

	missings_assess_train.index.name = 'training_data'
	print(missings_assess_train.head(10))
	print('\n')

	# Dataframe with the number of missings by feature (test data):
	missings_dict = df_test.isnull().sum().sort_values(ascending=False).to_dict()

	missings_assess_test = pd.DataFrame(data={
	    'feature': list(missings_dict.keys()),
	    'missings': list(missings_dict.values())
	})

	print('\033[1mNumber of features with missings:\033[0m {}'.format(sum(missings_assess_test.missings > 0)) +
	      ' out of {} features'.format(len(missings_assess_test)) +
	      ' ({}%).'.format(round((sum(missings_assess_test.missings > 0)/len(missings_assess_test))*100, 2)))
	print('\033[1mAverage number of missings:\033[0m {}'.format(int(missings_assess_test.missings.mean())) +
	      ' out of {} observations'.format(len(df_test)) +
	      ' ({}%).'.format(round((int(missings_assess_test.missings.mean())/len(df_test))*100,2)))
	print('\n')
	missings_assess_test.index.name = 'test_data'
	print(missings_assess_test.head(10))
	print('\n')

####################################################################################################################################
# Logarithmic transformation:

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mAPPLYING LOGARITHMIC TRANSFORMATION OVER NUMERICAL DATA\033[0m')
	print('\n')
	# Variables that should not be log-transformed:
	not_log = [c for c in df_train.columns if c not in cont_vars]

	if log_transform:
	    print('\033[1mTraining data:\033[0m')

	    # Assessing missing values (before logarithmic transformation):
	    num_miss_train = df_train.isnull().sum().sum()
	    if num_miss_train > 0:
	        print('\033[1mNumber of overall missings detected (before logarithmic transformation):\033[0m ' +
	              str(num_miss_train) + '.')

	    log_transf = log_transformation(not_log=not_log)
	    log_transf.transform(df_train)
	    df_train = log_transf.log_transformed

	    # Assessing missing values (after logarithmic transformation):
	    num_miss_train_log = df_train.isnull().sum().sum()
	    if num_miss_train_log > 0:
	        print('\033[1mNumber of overall missings detected (after logarithmic transformation):\033[0m ' + 
	              str(num_miss_train_log) + '.')

	    # Checking consistency in the number of missings:
	    if num_miss_train_log != num_miss_train:
	        print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	    print('\n')
	    print('\033[1mTest data:\033[0m')

	    # Assessing missing values (before logarithmic transformation):
	    num_miss_test = df_test.isnull().sum().sum()
	    if num_miss_test > 0:
	        print('\033[1mNumber of overall missings detected (before logarithmic transformation):\033[0m ' +
	              str(num_miss_test) + '.')

	    log_transf = log_transformation(not_log=not_log)
	    log_transf.transform(df_test)
	    df_test = log_transf.log_transformed

	    # Assessing missing values (after logarithmic transformation):
	    num_miss_test_log = df_test.isnull().sum().sum()
	    if num_miss_test_log > 0:
	        print('\033[1mNumber of overall missings detected (after logarithmic transformation):\033[0m ' + 
	              str(num_miss_test_log) + '.')

	    # Checking consistency in the number of missings:
	    if num_miss_test_log != num_miss_test:
	        print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	else:
	    print('\033[1mNo transformation performed!\033[0m')

	print('\n')
	print('---------------------------------------------------------------------------------------------------------')
	print('\n')

####################################################################################################################################
# Standardizing numerical features:

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mAPPLYING STANDARD SCALE TRANSFORMATION OVER NUMERICAL DATA\033[0m')
	print('\n')
	# Inputs that should not be standardized:
	not_stand = [c for c in df_train.columns if c.replace('L#', '') not in cont_vars]

	if standardize:
	    print('\033[1mTraining data:\033[0m')

	    stand_scale = standard_scale(not_stand = not_stand)
	    
	    stand_scale.scale(train = df_train, test = df_test)
	    
	    df_train_scaled = stand_scale.train_scaled
	    print('\033[1mShape of df_train_scaled (after scaling):\033[0m ' + str(df_train_scaled.shape) + '.')

	    # Assessing missing values (after standardizing numerical features):
	    num_miss_train = df_train.isnull().sum().sum()
	    num_miss_train_scaled = df_train_scaled.isnull().sum().sum()
	    if num_miss_train_scaled > 0:
	        print('\033[1mNumber of overall missings:\033[0m ' + str(num_miss_train_scaled) + '.')
	    else:
	        print('\033[1mNo missing values detected (training data)!\033[0m')

	    if num_miss_train_scaled != num_miss_train:
	        print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')
	    
	    print('\n')
	    print('\033[1mTest data:\033[0m')
	    df_test_scaled = stand_scale.test_scaled
	    print('\033[1mShape of df_test_scaled (after scaling):\033[0m ' + str(df_test_scaled.shape) + '.')

	    # Assessing missing values (after standardizing numerical features):
	    num_miss_test = df_test.isnull().sum().sum()
	    num_miss_test_scaled = df_test_scaled.isnull().sum().sum()
	    if num_miss_test_scaled > 0:
	        print('\033[1mNumber of overall missings:\033[0m ' + str(num_miss_test_scaled) + '.')
	    else:
	        print('\033[1mNo missing values detected (test data)!\033[0m')

	    if num_miss_test_scaled != num_miss_test:
	        print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	else:
	    df_train_scaled = df_train.copy()
	    df_test_scaled = df_test.copy()
	    
	    print('\033[1mNo transformation performed!\033[0m')

	print('\n')
	print('---------------------------------------------------------------------------------------------------------')
	print('\n')

####################################################################################################################################
# Treating missing values:

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mTREATING MISSING VALUES\033[0m')
	print('\n')

	print('\033[1mTraining data:\033[0m')
	num_miss_train = df_train_scaled.isnull().sum().sum()
	print('\033[1mNumber of overall missing values detected before treatment:\033[0m ' +
	      str(num_miss_train) + '.')

	# Loop over features:
	for f in df_train_scaled.drop(drop_vars, axis=1):
	    # Checking if there is missing values for a given feature:
	    if df_train_scaled[f].isnull().sum() > 0:
	        check_missing = impute_missing(df_train_scaled[f])
	        df_train_scaled[f] = check_missing['var']
	        df_train_scaled['NA#' + f.replace('L#', '')] = check_missing['missing_var']

	num_miss_train_treat = int(sum([sum(df_train_scaled[f]) for f in df_train_scaled.columns if 'NA#' in f]))
	print('\033[1mNumber of overall missing values detected during treatment:\033[0m ' +
	      str(num_miss_train_treat) + '.')

	if num_miss_train_treat != num_miss_train:
	    print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	if df_train_scaled.isnull().sum().sum() > 0:
	    print('\033[1mProblem - Number of overall missings detected (training data):\033[0m ' +
	          str(df_train_scaled.isnull().sum().sum()) + '.')

	print('\n')
	print('\033[1mTest data:\033[0m')
	num_miss_test = df_test_scaled.isnull().sum().sum()
	num_miss_test_treat = 0
	print('\033[1mNumber of overall missing values detected before treatment:\033[0m ' + str(num_miss_test) + '.')

	# Loop over features:
	for f in df_test_scaled.drop(drop_vars, axis=1):
	    # Check if there is dummy variable of missing value status for training data:
	    if 'NA#' + f.replace('L#', '') in list(df_train_scaled.columns):
	        check_missing = impute_missing(df_test_scaled[f])
	        df_test_scaled[f] = check_missing['var']
	        df_test_scaled['NA#' + f.replace('L#', '')] = check_missing['missing_var']
	    else:
	        # Checking if there are missings for variables without missings in training data:
	        if df_test_scaled[f].isnull().sum() > 0:
	            num_miss_test_treat += df_test_scaled[f].isnull().sum()
	            df_test_scaled[f].fillna(0, axis=0, inplace=True)

	num_miss_test_treat += int(sum([sum(df_test_scaled[f]) for f in df_test_scaled.columns if 'NA#' in f]))
	print('\033[1mNumber of overall missing values detected during treatment:\033[0m ' +
	      str(num_miss_test_treat) + '.')

	if num_miss_test_treat != num_miss_test:
	    print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	if df_test_scaled.isnull().sum().sum() > 0:
	    print('\033[1mProblem - Number of overall missings detected (test data):\033[0m ' +
	          str(df_test_scaled.isnull().sum().sum()) + '.')

####################################################################################################################################
# Transforming categorical features:

	# Creating dummies through one-hot encoding:
	# Create object for one-hot encoding:
	categorical_transf = one_hot_encoding(categorical_features = cat_vars)

	# Creating dummies:
	categorical_transf.create_dummies(categorical_train = categorical_train,
	                                  categorical_test = categorical_test)

	# Selected dummies:
	dummy_vars = list(categorical_transf.dummies_train.columns)

	# Training data:
	dummies_train = categorical_transf.dummies_train
	dummies_train.index = df_train_scaled.index

	# Test data:
	dummies_test = categorical_transf.dummies_test
	dummies_test.index = df_test_scaled.index

	# Dropping original categorical features:
	df_train_scaled.drop(cat_vars, axis=1, inplace=True)
	df_test_scaled.drop(cat_vars, axis=1, inplace=True)

	print('\033[1mNumber of categorical features:\033[0m {}.'.format(len(categorical_transf.categorical_features)))
	print('\033[1mNumber of overall selected dummies:\033[0m {}.'.format(dummies_train.shape[1]))
	print('\033[1mShape of dummies_train for store ' + str(s) + ':\033[0m ' +
	      str(dummies_train.shape) + '.')
	print('\033[1mShape of dummies_test for store ' + str(s) + ':\033[0m ' +
	      str(dummies_test.shape) + '.')
	print('\n')

	# Concatenating all features:
	df_train_scaled = pd.concat([df_train_scaled, dummies_train], axis=1)
	df_test_scaled = pd.concat([df_test_scaled, dummies_test], axis=1)

	print('\033[1mShape of df_train_scaled for store ' + str(s) + ':\033[0m ' + str(df_train_scaled.shape) + '.')
	print('\033[1mShape of df_test_scaled for store ' + str(s) + ':\033[0m ' + str(df_test_scaled.shape) + '.')
	print('\n')

	# Assessing missing values (training data):
	num_miss_train = df_train_scaled.isnull().sum().sum() > 0
	if num_miss_train:
	    print('\033[1mProblem - Number of overall missings detected (training data):\033[0m ' +
	          str(df_train_scaled.isnull().sum().sum()) + '.')
	    print('\n')

	# Assessing missing values (test data):
	num_miss_test = df_test_scaled.isnull().sum().sum() > 0
	if num_miss_test:
	    print('\033[1mProblem - Number of overall missings detected (test data):\033[0m ' +
	          str(df_test_scaled.isnull().sum().sum()) + '.')
	    print('\n')

####################################################################################################################################
# Datasets structure:

	# Checking consistency of structure between training and test dataframes:
	if len(list(df_train_scaled.columns)) != len(list(df_test_scaled.columns)):
	    print('\033[1mProblem - Inconsistent number of columns between dataframes for training and test data!\033[0m')

	else:
	    consistency_check = 0
	    
	    # Loop over variables:
	    for c in list(df_train_scaled.columns):
	        if list(df_train_scaled.columns).index(c) != list(df_test_scaled.columns).index(c):
	            print('\033[1mProblem - Feature {0} was positioned differently in training and test dataframes!\033[0m'.format(c))
	            consistency_check += 1
	            
	    # Reordering columns of test dataframe:
	    if consistency_check > 0:
	        ordered_columns = list(df_train_scaled.columns)
	        df_test_scaled = df_test_scaled[ordered_columns]

	print('\n')
	print('---------------------------------------------------------------------------------------------------------')
	print('\n')

	print('******************************************************************************************')
	print('\033[1mSTAGE OF DATA PRE-PROCESSING: finished!\033[0m')
	print('******************************************************************************************')
	print('\n')

####################################################################################################################################
####################################################################################################################################
######################################################DATA MODELING#################################################################

	print('******************************************************************************************')
	print('\033[1mSTAGE OF DATA MODELING\033[0m')
	print('******************************************************************************************')
	print('\n')

	print('-----------------------------------------------')
	print('\033[1mTasks:\033[0m')
	print('Performing grid search to define hyper-parameters using training data and evaluating performance metrics based on test data.')
	print('\033[1mEstimation method:\033[0m ' + method.replace('_', ' ') + '.')
	print('-----------------------------------------------')
	print('\n')

####################################################################################################################################
# Grids of hyper-parameters:

	# Declare grid of hyper-parameters:
	if method == 'logistic_regression':
	    # Default values for hyper-parameters:
	    params_default = {'C': 1}

	    # Grid of values for hyper-parameters:
	    grid_regul = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.25, 0.3, 0.5, 0.75, 1, 3, 10]

	    if random_search:
	        # Number of samples from each random distribution of regularization parameter:
	        n1 = int(n_samples*round(len([x for x in grid_regul if (x < 0.1)])/len(grid_regul), 2))
	        n2 = int(n_samples*round(len([x for x in grid_regul if (x >= 0.1) & (x < 1)])/len(grid_regul), 2)) + 1
	        n3 = int(n_samples*round(len([x for x in grid_regul if x >= 1])/len(grid_regul), 2))

	        grid_regul = []

	        # Loop over random distributions of regularization parameter:
	        for d in [uniform(0.0001, 0.1).rvs(n1), uniform(0.1, 1).rvs(n2), uniform(1, 10).rvs(n3)]:
	            for x in d:
	                grid_regul.append(x)
	    
	    params = {'C': grid_regul}

	elif method == 'GBM':
	    # Default values of hyper-parameters:
	    params_default = {'subsample': 0.75,
	                      'learning_rate': 0.01,
	                      'max_depth': 3,
	                      'n_estimators': 500}
	    
	    if random_search:
	        # Random distributions of hyper-parameters:
	        params = {'subsample': [0.75],
	                  'learning_rate': uniform(0.0001, 0.1),
	                  'max_depth': randint(1, 5+1),
	                  'n_estimators': randint(100, 500+1)}

	    else:
	        # Grid of values for hyper-parameters:
	        params = {'subsample': [0.75],
	                  'learning_rate': [0.0001, 0.01, 0.1],
	                  'max_depth': [1, 3, 5],
	                  'n_estimators': [100, 250, 500]}

####################################################################################################################################
# Estimations:

	# Declare K-folds CV estimation object:
	train_test_est = Kfolds_fit(task = 'classification', method = method,
	                            metric = 'roc_auc', num_folds = 3,
	                            pre_selecting = False, pre_selecting_param = None,
	                            random_search = random_search, n_samples = n_samples,
	                            grid_param = params, default_param = params_default)

	# Running train-test estimation:
	train_test_est.run(train_inputs=df_train_scaled.drop(drop_vars, axis=1),
	                   train_output=df_train_scaled['y'],
	                   test_inputs=df_test_scaled.drop(drop_vars, axis=1),
	                   test_output=df_test_scaled['y'])

	# Defining best tuning hyper-parameter:
	best_params = train_test_est.best_param

	# Assessing performance metrics:
	test_roc_auc = train_test_est.performance_metrics["test_roc_auc"]
	test_prec_avg = train_test_est.performance_metrics["test_prec_avg"]
	test_brier = train_test_est.performance_metrics["test_brier"]

	print('******************************************************************************************')
	print('\033[1mSTAGE OF DATA MODELING: finished!\033[0m')
	print('******************************************************************************************')
	print('\n')

####################################################################################################################################
####################################################################################################################################
######################################################ASSESSMENT OF RESULTS#########################################################

	print('******************************************************************************************')
	print('\033[1mSTAGE OF ASSESSMENT OF RESULTS\033[0m')
	print('******************************************************************************************')
	print('\n')

	print('-------------------------------')
	print('\033[1mTasks:\033[0m')
	print('Collecting performance metrics.')
	print('-------------------------------')
	print('\n')

	end_time_store = datetime.now()

	model_assessment[estimation_id] = {
	    'store_id': s,
	    'n_orders_train': len(df_train_scaled),
	    'n_orders_test': len(df_test_scaled),
	    'n_vars': df_train_scaled.drop(drop_vars, axis=1).shape[1],
	    'first_date_train': str(df_train_scaled.date.min().date()),
	    'last_date_train': str(df_train_scaled.date.max().date()),
	    'first_date_test': str(df_test_scaled.date.min().date()),
	    'last_date_test': str(df_test_scaled.date.max().date()),
	    'avg_order_amount_train': df_train_scaled.order_amount.mean(),
	    'avg_order_amount_test': df_test_scaled.order_amount.mean(),
	    'log_transform': log_transform,
	    'standardize': standardize,
	    'method': method,
	    'random_search': random_search,
	    'n_samples': n_samples,
	    'best_param': str(best_params),
	    'test_roc_auc': test_roc_auc,
	    'test_prec_avg': test_prec_avg,
	    'test_brier': test_brier,
	    'running_time': str(round((end_time_store - start_time_store).seconds/60 , 2)) + ' minutes'
	}

	os.chdir('/home/matheus_rosso/Arquivo/Materiais/Codes/grid_random_searches/')

	if export:
	    with open('Datasets/model_assessment.json', 'w') as json_file:
	        json.dump(model_assessment, json_file, indent=2)

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
	with open('last_estimation.json', 'w') as json_file:
		json.dump({'last_estimation': str(S.index(s)+1) + ' out of ' + str(len(S))}, json_file, indent=2)

# Assessing overall running time:
end_time_all = datetime.now()

print('\n')
print('------------------------------------')
print('\033[1mOverall running time:\033[0m ' + str(round(((end_time_all - start_time_all).seconds)/60, 2)) + ' minutes.')
print('Start time: ' + start_time_all.strftime('%Y-%m-%d') + ', ' + start_time_all.strftime('%H:%M:%S'))
print('End time: ' + end_time_all.strftime('%Y-%m-%d') + ', ' + end_time_all.strftime('%H:%M:%S'))
print('\n')
