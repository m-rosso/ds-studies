####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

####################################################################################################################################
# Logarithm of selected features:

class log_transformation(object):
    """Applies function to log-transform all variables in a dataframe except for those
    explicitly declared. Returns the dataframe with selected variables log-transformed
    and their respective names changed to 'L#PREVIOUS_NAME()'."""

    def __init__(self, not_log):
        self.not_log = not_log
        
    def transform(self, data):
        # Function that applies natural logarithm to numerical variables:
        def log_func(x):
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
        
        # Redefining names of columns:
        new_col = []
        log_vars = []
        
        self.log_transformed = data
        
        # Applying logarithmic transformation to selected variables:
        for f in list(data.columns):
            if f in self.not_log:
                new_col.append(f)
            else:
                new_col.append('L#' + f)
                log_vars.append('L#' + f)
                self.log_transformed[f] = data[f].apply(log_func)

        self.log_transformed.columns = new_col
        
        print('\033[1mNumber of numerical variables log-transformed:\033[0m ' + str(len(log_vars)) + '.')

####################################################################################################################################
# Standardizing selected features:

class standard_scale(object):
    """Fits and transforms all variables in a dataframe, except for those explicitly defined to not scale.
    Uses 'StandardScaler' from sklearn and returns not only scaled data, but also in its dataframe original
    format. If test data is provided, then their values will be standardized using means and variances from
    train data."""
    
    def __init__(self, not_stand):
        self.not_stand = not_stand
    
    def scale(self, train, test=None):
        # Creating standardizing object:
        scaler = StandardScaler()
        
        # Calculating means and variances:
        scaler.fit(train.drop(self.not_stand, axis=1))
        
        # Standardizing selected variables:
        self.train_scaled = scaler.transform(train.drop(self.not_stand, axis=1))
        
        # Transforming data into dataframe and concatenating selected and non-selected variables:
        self.train_scaled = pd.DataFrame(data=self.train_scaled,
                                         columns=train.drop(self.not_stand, axis=1).columns)
        self.train_scaled.index = train.index
        self.train_scaled = pd.concat([train[self.not_stand], self.train_scaled], axis=1)
        
        # Test data:
        if test is not None:
            # # Standardizing selected variables:
            self.test_scaled = scaler.transform(test.drop(self.not_stand, axis=1))
            
            # Transforming data into dataframe and concatenating selected and non-selected variables:
            self.test_scaled = pd.DataFrame(data=self.test_scaled,
                                            columns=test.drop(self.not_stand, axis=1).columns)
            self.test_scaled.index = test.index
            self.test_scaled = pd.concat([test[self.not_stand], self.test_scaled], axis=1)
