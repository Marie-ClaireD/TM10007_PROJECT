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
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix, roc_auc_score
from scipy import interp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from pprint import pprint
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


#%%
#%%


from hn.load_data import load_data

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

#%%
def support_vector(x, y):
    """ 
    Support vector machine
    """
    crss_val = RepeatedKFold(n_splits = 5, n_repeats=1, random_state = None)           
    crss_val.get_n_splits(x, y)

    accuracy_rs = []
    accuracies = []
    auc_scores = []
    specificities = []
    sensitivities = []
    tprs = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)
    #predict_labels = []
    #predict_probas = []
    #y_val_total = []
    param_dist = {"C": randint(0.1, 100),
                  "gamma": ['auto','scale'],
                  "kernel": ['rbf','poly','sigmoid','linear']}
    clf = SVC(probability=True) 
    #idx = np.arange(0, len(y))
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1) #Hier nog een keer CV?
    model = random_search.fit(x, y)
    hyperparameters = model.best_estimator_.get_params()
    print(hyperparameters)

    for train_index, val_index in crss_val.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val= y[train_index], y[val_index]
        
        clf = SVC(C=hyperparameters.get("C"), gamma=hyperparameters.get("gamma"), kernel=hyperparameters.get("kernel"), probability=True)
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_val)        
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

        predicted_probas = random_search.predict_proba(x_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, predicted_probas)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    mean_auc = auc(base_fpr, mean_tprs)
    std_auc = np.std(aucs)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    plt.figure(figsize=(12, 8))
    plt.plot(base_fpr, mean_tprs, 'c', alpha=0.8, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='c', alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.grid()
    plt.show()

    return performance_scores

performances = support_vector(x_train, y_train)
print(performances)



# %%
