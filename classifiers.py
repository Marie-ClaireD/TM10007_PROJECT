#%%
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix, roc_auc_score
from numpy import interp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from hn.load_data import load_data
from pprint import pprint

#%%
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

#%%

""" Extract feature values and labels """
# Features
features = data.loc[:, data.columns != 'label'].values
features = StandardScaler().fit_transform(features)

# Labels
labels = data.loc[:,['label']].values
labels = [item if item!='T12' else 0 for item in labels]
labels = [item if item!='T34' else 1 for item in labels]
labels = np.array(labels)
print(f'Number of high risk patients: {np.count_nonzero(labels)}') 
print(f'Number of low risk patients: {len(labels) - np.count_nonzero(labels)}')

#%%

def split_sets(features, labels):
    """
    Splits the features and labels into a training set (80%) and test set (20%)
    """
    x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test 

x_train, x_test, y_train, y_test = split_sets(features, labels) 

# %%
def get_hyperparameters(x, y):
    """ 
    Random Search for Hyperparameters classifiers
    """
    
    clsfs = [KNeighborsClassifier(), RandomForestClassifier(bootstrap=True, random_state=None), SVC(probability=True)]
    param_distributions = [{"n_neighbors": randint(1, 20)}, {"n_estimators": randint(1, 200),
                "max_features": randint(5, 30),
                "max_depth": randint(2, 18),
                "min_samples_leaf": randint(1, 17)},{"C": randint(0.1, 100),
                 "gamma": ['auto','scale'],
                 "kernel": ['rbf','poly','sigmoid','linear']}]

    hyperparameters_clsfs = []
    for clf, param_dist in zip(clsfs, param_distributions):
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, n_jobs=-1) #Hier nog een keer CV?
        model = random_search.fit(x, y)
        parameters = model.best_estimator_.get_params()
        pprint(parameters)
        hyperparameters_clsfs.append(parameters)

    return hyperparameters_clsfs

hyperparameters = get_hyperparameters(x_train, y_train)
print(hyperparameters)

#%% Cross Validation Classifiers
def cross_validation(x, y, hyperparameters):
    """ 
    Random Forest Random Search for Hyperparameters
    """

    crss_val = RepeatedKFold(n_splits=5, n_repeats=5, random_state=None)           
    crss_val.get_n_splits(x, y)

    accuracies = []
    auc_scores = []
    specificities = []
    sensitivities = []

    dictionary = {}
    clsfs = [LogisticRegression(), KNeighborsClassifier(n_neighbors=hyperparameters[0].get('n_neighbors')), RandomForestClassifier(bootstrap=True, max_depth=hyperparameters[1].get('max_depth'), max_features=hyperparameters[1].get('max_features'), min_samples_leaf=hyperparameters[1].get('min_samples_leaf'), n_estimators=hyperparameters[1].get('n_estimators'), random_state=None), SVC(C=hyperparameters[2].get("C"), gamma=hyperparameters[2].get("gamma"), kernel=hyperparameters[2].get("kernel"), probability=True)]

    for clf in clsfs:
        
        for train_index, val_index in crss_val.split(x, y):

            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            if min(x_train.shape[0], x_train.shape[1]) < 70:
                print('Not enough input values for PCA with 70 components')
                sys.exit()

            else:
                pca = PCA(n_components=70)
                pca.fit(x_train)
                x_train = pca.transform(x_train)
                x_val = pca.transform(x_val)
        
                clf.fit(x_train, y_train)
                prediction = clf.predict(x_val)
                predicted_probas = clf.predict_proba(x_val)[:, 1]

                performance_scores = pd.DataFrame()                    
                auc_scores.append(roc_auc_score(y_val, prediction))
                conf_mat = confusion_matrix(y_val, prediction)
                total = sum(sum(conf_mat))
                accuracies.append((conf_mat[0, 0]+conf_mat[1, 1])/total)
                sensitivities.append(conf_mat[0, 0]/(conf_mat[0, 0]+conf_mat[0, 1]))
                specificities.append(conf_mat[1, 1]/(conf_mat[1, 0]+conf_mat[1, 1]))
                performance_scores['Accuracy'] = accuracies
                performance_scores['AUC'] = auc_scores
                performance_scores['Sensitivity'] = sensitivities
                performance_scores['Specificity'] = specificities

    dictionary[clf] = performance_scores
    
    return dictionary


performances = cross_validation(x_train, y_train, hyperparameters)
print(performances)
   


# %%

    


  


