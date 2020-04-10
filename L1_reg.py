""" L1 regularization """
# General Packages
import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from scipy import interp
from scipy.stats import randint

# Load Data
from hn.load_data import load_data

# Classifiers & Kernels
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Pre-processing
from sklearn.preprocessing import StandardScaler

# Feature Selection
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel

# Model Selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

# Evaluation
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix

def load_check_data():
    '''
    Check if the datafile exists and is valid before reading. Impute missing data.
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
    # Impute missing or NaN datapoints are with the average of that feature.
    if data.isnull().values.any():
        column_mean = data.mean()
        data = data.fillna(column_mean)
        print('In the csv data file, some values are missing or NaN.'
              'These missing values are replaced by the mean of that feature.')
    return data
data = load_check_data()

# %% Extract feature values and labels
# Extract features from data
features = data.loc[:, data.columns != 'label'].values

# Extract labels from data
labels = data.loc[:,['label']].values

# Low risk patients receive the label 0 and high risk the label 1
labels = [item if item!='T12' else 0 for item in labels]
labels = [item if item!='T34' else 1 for item in labels]
labels = np.array(labels)

# Number of high and low risk patients is printed
print(f'Number of high risk patients: {np.count_nonzero(labels)}')
print(f'Number of low risk patients: {len(labels) - np.count_nonzero(labels)}')

# %% Scale and split data into training, validation and test set
def scale_data(x, y):
    """
    Scale data with Standard scaler
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)
    x_test = scaler.transform(y)
    return x_train, x_test
    
def split_sets(x, y):
    """
    Splits the features and labels into a training set (80%) and test set (20%).
    Splitting in the train and test set is shown to provide insight into our method 
    and is used to compute the principal components and alpha for LASSO estimator. 
    """
    # Use repeated stratified KFold
    crss_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    
    # Splitting data into test and training set
    for train_index, test_index in crss_val.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Scale the data by fitting on the training set and transforming the test set
        x_train, x_test = scale_data(x_train, x_test)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = split_sets(features, labels)


def get_best_alpha(x1, y1, x2, y2):  
    """ 
    Tuning alpha parameter for optimal feature selection with LASSO
    """
    
    coefs = []
    accuracies = []
    training_scores = []
    test_scores = []
    a_max=0
    n_alphas = 100
    alphas = np.logspace(-2,0,num=n_alphas)
    for a in alphas:
        # Fit classifier
        clf = Lasso(alpha=a, random_state=None)
        clf.fit(pd.DataFrame(x1).fillna(0), y1)
        coeff_used = np.sum(clf.coef_ != 0) # Number of coefficents with non zero weight
 
        # Compute determination coefficient and selecting best alpha
        training_score = (clf.score(x1,y1))
        coeff_used = np.sum(clf.coef_ != 0) # Number of coefficents with non zero weight
        test_score = clf.score(x2, y2)
        if test_score > a_max:
            a_max = test_score
            best_alpha = a

    return best_alpha, a_max

best_alpha, a_max = get_best_alpha(x_train, y_train, x_test, y_test)
print(a_max)
print(best_alpha)

alphas1 = [0.03853528593710529, 0.04229242874389499, 0.03853528593710529, 0.040370172585965536, 0.040370172585965536]
hoi = np.mean(alphas1)
print(hoi)

# %% Finding best value for alpha parameter (Lasso)





# %% Perform L1 regularization using Linear regression (Lasso)
#lasso = SelectFromModel(estimator=Lasso(alpha=0.0335, random_state=None, max_iter=10000), threshold='median')
#lasso.fit(scaler.transform(pd.DataFrame(x_train).fillna(0)), y_train)
#lasso.get_support()

#selected_feat = data.columns[(lasso.get_support())]
#print('total features: {}'.format((x_train.shape[1])))
#print('features with coefficients shrank to zero: {}'.format(
#     np.sum(lasso.estimator_.coef_ == 0)))
#print('Feature coefficients:',lasso.estimator_.coef_)

# Getting a list of removed features
#removed_feats = data.columns[(lasso.estimator_.coef_ == 0).ravel().tolist()]
#print(removed_feats)

# Remove features from training and test set
#x_train_selected = lasso.transform(pd.DataFrame(x_train).fillna(0))
#x_test_selected = lasso.transform(pd.DataFrame(x_test).fillna(0))
 
#print(x_train_selected.shape, x_test_selected.shape)


#n_alphas = 100
#alphas = np.logspace(-2,0,num=n_alphas)

#####  Los berekenen van scores per alpha ###### 
# Construct classifiers
#coefs = []
#accuracies = []
#training_scores = []
#test_scores = []
#a_max=0
#for a in alphas:
    # Fit classifier
 #   clf = Lasso(alpha=a, max_iter=1000, random_state=None)
  #  clf.fit(pd.DataFrame(x_train).fillna(0)), y_train)
  #  training_score = (clf.score(x_train,y_train))
   # test_score = clf.score(x_test,y_test)
  #  coeff_used = np.sum(clf.coef_ != 0) # Number of coefficents with non zero weight

 #   if test_score > a_max: 
  #      a_max = test_score
  #      best_alpha = a
 
    # Zet dit uit als je het niet voor elke wil printen! 
    #print('For alpha =',a)
    #print('training score:',training_score)
    #print('testing score:',test_score)
    #print('number of features used:',coeff_used)



## dit niet meer nodig 

# %% Univariate feature selection
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_classif

# feature extraction
#univariate = SelectKBest(score_func=f_classif, k=15)
#univariate.fit(scaler.transform(pd.DataFrame(x_train).fillna(0)), y_train)
#univariate.get_support()
#selected_feat = data.columns[(univariate.get_support())]
#print('total features: {}'.format((x_train.shape[1])))
#print('selected features with Univariate testing: {}'.format(len(selected_feat)))

# %% Remove features from training and test set
#x_train_selected = univariate.transform(pd.DataFrame(x_train).fillna(0))
#print(x_train_selected.shape)
