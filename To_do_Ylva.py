import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix, roc_auc_score
from numpy import interp


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


""" Extract feature values and labels """
# Features
features = data.loc[:, data.columns != 'label'].values
features = StandardScaler().fit_transform(features)

# Labels
labels = data.loc[:, ['label']].values
labels = [item if item != 'T12' else 0 for item in labels]
labels = [item if item != 'T34' else 1 for item in labels]
labels = np.array(labels)
print(f'Number of high risk patients: {np.count_nonzero(labels)}') 
print(f'Number of low risk patients: {len(labels) - np.count_nonzero(labels)}')

# The training set is again divided into a training and a validation set and afterwards
# classified using a Kfold cross validation and logistic regression.


def split_sets(x, y):
    '''
    Splits the features and labels into a training set (80%) and test set (20%)
    '''
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test 


x_train, x_test, y_train, y_test = split_sets(features, labels)


def cross_val(x, y):
    '''
    Cross validation using a Logistic Regression classifier (5 folds)
    '''

    crss_val = RepeatedKFold(n_splits=5, n_repeats=100, random_state=None)        
    crss_val.get_n_splits(x, y)

    accuracies = []
    auc_scores = []
    tprs = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)

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
            lrg = LogisticRegression()
            lrg.fit(x_train, y_train)
            prediction = lrg.predict(x_val)
                                
            accuracies.append(accuracy_score(y_val, prediction))
            auc_scores.append(roc_auc_score(y_val, prediction))
            performance_scores = pd.DataFrame()
            performance_scores['Accuracy'] = accuracies
            performance_scores['AUC'] = auc_scores

            predicted_probas = lrg.predict_proba(x_val)[:, 1]
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


performances = cross_val(x_train, y_train)


