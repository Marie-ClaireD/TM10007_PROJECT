#%%
# L1 regularization
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix
from scipy import interp
from hn.load_data import load_data
import seaborn as sns 
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

def load_check_data():
    '''
    Check if the datafile exists and is valid before reading
    '''
    # Check whether datafile exists
    try:
        data = load_data()
        print(f'The number of samples: {len(data.index)}')
        print(f'The number of columns: {len(data.columns)}')
    except FileNotFoundError:
        return print("The csv datafile does not exist"), sys.exit()
    except pd.errors.ParserError:
        return print('The csv datafile is not a proper csv format.'
                     'Please provide a data file in csv format.'), sys.exit()
    # Check whether data is missing.
    # If any datapoints are missing or NaN, these empty cells are replaced with the average 
    # of that feature.
    if data.isnull().values.any():
        column_mean = data.mean()
        data = data.fillna(column_mean)
        print('In the csv data file, some values are missing or NaN.'
              ' These missing values are replaced by the mean of that feature.')
    return data
data = load_check_data()
data.shape

numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]
data.shape

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def split_sets(x, y):
    '''
    Splits the features and labels into a training set (80%) and test set (20%)
    '''
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=None)
    return x_train, x_test, y_train, y_test
    

x_train, x_test, y_train, y_test = split_sets(features, labels)
scaler = StandardScaler()
scaler.fit(pd.DataFrame(x_train).fillna(0))

sel_ = SelectFromModel(LogisticRegression(solver='saga', C=1, penalty='l1'))
sel_.fit(scaler.transform(pd.DataFrame(x_train).fillna(0)), y_train)
sel_.get_support()

selected_feat = data.columns[(sel_.get_support())]
print('total features: {}'.format((x_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0)))



removed_feats = data.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
removed_feats

# %%
x_train_selected = sel_.transform(pd.DataFrame(x_train).fillna(0))

x_test_selected = sel_.transform(pd.DataFrame(x_test).fillna(0))


