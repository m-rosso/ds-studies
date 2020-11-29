####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np

from datetime import datetime
import time

####################################################################################################################################
####################################################################################################################################
#############################################################FUNCTIONS##############################################################

# Function that converts epoch into date:
def epoch_to_date(x):
    str_datetime = time.strftime('%d %b %Y', time.localtime(x/1000))
    dt = datetime.strptime(str_datetime, '%d %b %Y')
    return dt

# Function that calculates binomial deviance for a set of data points:
def binomial_deviance(y, p):
    "y is a true binary label, while p is an estimated probability for reference class."
    return np.sum(np.log(1 + np.exp(-2*y*p)))
