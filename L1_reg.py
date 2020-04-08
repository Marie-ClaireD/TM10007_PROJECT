""" L1 regularization """
# %% Import modules
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import interp
from hn.load_data import load_data
from scipy.stats import randint
from pprint import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, StratifiedKFold 
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, LogisticRegression, LassoCV
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# %% Load and check data
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

# %% Extract feature values and labels
features = data.loc[:, data.columns != 'label'].values
labels = data.loc[:, ['label']].values
labels = [item if item != 'T12' else 0 for item in labels]
labels = [item if item != 'T34' else 1 for item in labels]
labels = np.array(labels)

numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]

# %% Scale and split data into training, validation and test set
def split_sets(x, y):
    '''
    Splits the features and labels into a training set (80%) and test set (20%)
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, x_test, y_train, y_test
    
x_train, x_test, y_train, y_test = split_sets(features, labels)
scaler = StandardScaler()
scaler.fit(pd.DataFrame(x_train).fillna(0))
scaler.transform(x_test)
# %% Perform L1 regularization using Logistic regression and L1 penalty
# sel_ = SelectFromModel(LogisticRegression(solver='saga', C=1, penalty='l1'))
# sel_.fit(scaler.transform(pd.DataFrame(x_train).fillna(0)), y_train)
# sel_.get_support()

# selected_feat = data.columns[(sel_.get_support())]
# print('total features: {}'.format((x_train.shape[1])))
# print('selected features: {}'.format(len(selected_feat)))
# print('features with coefficients shrank to zero: {}'.format(
#      np.sum(sel_.estimator_.coef_ == 0)))

# %% Finding best value for alpha parameter (Lasso)
n_alphas = 200
alphas = np.logspace(-9,1,num=n_alphas)

#####  Los berekenen van scores per alpha ###### 
# Construct classifiers
coefs = []
accuracies = []
training_scores = []
test_scores = []

for a in alphas:
    # Fit classifier
    clf = Lasso(alpha=a, max_iter=10000)
    clf.fit(scaler.transform(pd.DataFrame(x_train).fillna(0)), y_train)
    training_score = (clf.score(x_train,y_train))
    test_score = clf.score(x_test,y_test)
    coeff_used = np.sum(clf.coef_ != 0) # Number of coefficents with non zero weight

    training_scores.append(training_score)
    test_scores.append(test_score)

    # Zet dit uit als je het niet voor elke wil printen! 
    #print('For alpha =',a)
    #print('training score:',training_score)
    #print('testing score:',test_score)
    #print('number of features used:',coeff_used)

#### Weights en accuracy's van verschillende alphas plotten
# Construct classifiers
coefs = []
accuracies = []

for a in alphas:
    # Fit classifier
    clf = Lasso(alpha=a, fit_intercept=False, max_iter = 10000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    # Append statistics
    accuracy = float((y_test == y_pred).sum()) / float(y_test.shape[0])
    accuracies.append(accuracy)
    coefs.append(clf.coef_)

# Display results

# Weights
plt.figure()
ax = plt.gca()
ax.plot(alphas, np.squeeze(coefs))
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# Performance
plt.figure()
ax = plt.gca()
ax.plot(alphas, accuracies)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('accuracies')
plt.title('Performance as a function of the regularization')
plt.axis('tight')
plt.show()
    
# %% Perform L1 regularization using Linear regression (Lasso)
lasso = SelectFromModel(estimator=Lasso(alpha=0.0000001, random_state=None, max_iter=10000), threshold='median')
lasso.fit(scaler.transform(pd.DataFrame(x_train).fillna(0)), y_train)
lasso.get_support()

selected_feat = data.columns[(lasso.get_support())]
print('total features: {}'.format((x_train.shape[1])))
print('selected features with Lasso: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
     np.sum(lasso.estimator_.coef_ == 0)))
print('Feature coefficients:',lasso.estimator_.coef_)

# Getting a list of removed features
removed_feats = data.columns[(lasso.estimator_.coef_ == 0).ravel().tolist()]
print(removed_feats)

# Remove features from training and test set
x_train_selected = lasso.transform(pd.DataFrame(x_train).fillna(0))
x_test_selected = lasso.transform(pd.DataFrame(x_test).fillna(0))
 
print(x_train_selected.shape, x_test_selected.shape)






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
