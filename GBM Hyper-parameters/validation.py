####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
import os
import json

import progressbar
from time import sleep

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, brier_score_loss

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

####################################################################################################################################
# K-folds cross-validation for grid-search of only one hyper-parameter:

class KfoldsCV(object):
    def __init__(self, method = 'logistic_regression', metric = 'roc_auc', num_folds = 3,
                 random_search = False, n_samples = None,
                 grid_param = None, default_param = None):
        self.method = str(method)
        self.metric = metric
        self.num_folds = int(num_folds)
        self.default_param = default_param
        
        if random_search is not True:
            list_param = [grid_param[k] for k in grid_param.keys()]
            list_param = [list(x) for x in np.array(np.meshgrid(*list_param)).T.reshape(-1,len(list_param))]
            self.grid_param = []
            for i in list_param:
                self.grid_param.append(dict(zip(grid_param.keys(), i)))
            
        else:
            self.grid_param = []

            for i in range(1, n_samples+1):
                list_param = []

                for k in grid_param.keys():
                    try:
                        list_param.append(grid_param[k].rvs(1)[0])
                    except:
                        list_param.append(np.random.choice(grid_param[k]))
                self.grid_param.append(dict(zip(grid_param.keys(), list_param)))
    
    def run(self, inputs, output):
        metric = {
            'roc_auc': roc_auc_score,
            'avg_precision_score': average_precision_score,
            'brier_loss': brier_score_loss
        }
        
        k = list(range(self.num_folds))
        k_folds_X = np.array_split(inputs, self.num_folds)
        k_folds_y = np.array_split(output, self.num_folds)
        
        self.CV_metric = pd.DataFrame()
        CV_scores = dict(zip([str(g) for g in self.grid_param],
                             [pd.DataFrame(data=[],
                                           columns=['cv_score']) for i in range(len(self.grid_param))]))

        bar_grid = progressbar.ProgressBar(maxval=len(self.grid_param), widgets=['\033[1mGrid estimation progress:\033[0m ',
                                                                                progressbar.Bar('-', '[', ']'), ' ',
                                                                                progressbar.Percentage()])
        bar_grid.start()
        
        for j in range(len(self.grid_param)):
            CV_metric_list = []
            try:
                for i in k:
                    # Train and validation split:
                    X_train = pd.concat([x for l,x in enumerate(k_folds_X) if (l!=i)], axis=0, sort=False)
                    y_train = pd.concat([x for l,x in enumerate(k_folds_y) if (l!=i)], axis=0, sort=False)

                    X_val = k_folds_X[i]
                    y_val = k_folds_y[i]

                    # Create the estimation object:
                    if self.method == 'logistic_regression':
                        model = LogisticRegression(solver='liblinear',
                                                   penalty = 'l1',
                                                   C = self.grid_param[j]['C'],
                                                   warm_start=True)
                        
                    elif self.method == 'GBM':
                        model = GradientBoostingClassifier(subsample = float(self.grid_param[j]['subsample']),
                                                           max_depth = int(self.grid_param[j]['max_depth']),
                                                           learning_rate = float(self.grid_param[j]['learning_rate']),
                                                           n_estimators = int(self.grid_param[j]['n_estimators']),
                                                           warm_start = True)                        

                    # Training the model:
                    model.fit(X_train, y_train)

                    # Predicting scores and calculating ROC-AUC statistics:
                    score_pred = [p[1] for p in model.predict_proba(X_val)]
                    CV_metric_list.append(metric[self.metric](y_val, score_pred))
                    
                    # Dataframes with CV scores:
                    ref = pd.DataFrame(data={'y_true': list(k_folds_y[i]),
                                             'cv_score': score_pred},
                                       index=list(k_folds_y[i].index))
                    CV_scores[str(self.grid_param[j])] = pd.concat([CV_scores[str(self.grid_param[j])],
                                                                    ref], axis=0, sort=False)
                
                # Dataframes with CV ROC-AUC statistics:
                self.CV_metric = pd.concat([self.CV_metric,
                                            pd.DataFrame(data={'tun_param': str(self.grid_param[j]),
                                                               'cv_' + self.metric: np.nanmean(CV_metric_list)},
                                                         index=[j])], axis=0, sort=False)

            except:
                print('\033[1mNot able to perform CV estimation with parameters ' +
                      str(self.grid_param[j]) + '!\033[0m')
                
                self.CV_metric = pd.concat([self.CV_metric,
                                            pd.DataFrame(data={'tun_param': self.grid_param[j],
                                                               'cv_' + self.metric: np.NaN},
                                                         index=[j])], axis=0, sort=False)

            bar_grid.update(j+1)
            sleep(0.1)
        
        # Best tuning parameters:
        try:
            if (self.metric == 'brier_loss') | (self.metric == 'mse'):
                self.best_param = self.CV_metric['cv_' + self.metric].idxmin()
            else:
                self.best_param = self.CV_metric['cv_' + self.metric].idxmax()
                
            self.best_param = self.grid_param[self.best_param]
            
        except:
            self.best_param = self.default_param
        
        # CV scores for best tuning parameter:
        try:
            self.CV_scores = CV_scores[str(self.best_param)]
        except:
            self.CV_scores = pd.DataFrame(data=[])
