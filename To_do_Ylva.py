# %%
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
import seaborn as sns
from hn.load_data import load_data

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

# Labels
labels = data.loc[:, ['label']].values
labels = [item if item != 'T12' else 0 for item in labels]
labels = [item if item != 'T34' else 1 for item in labels]
labels = np.array(labels)
print(f'Number of high risk patients: {np.count_nonzero(labels)}')
print(f'Number of low risk patients: {len(labels) - np.count_nonzero(labels)}')

#%%
# # The training set is again divided into a training and a validation set and afterwards
# classified using a Kfold cross validation and logistic regression.

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

#%%
def get_hyperparameters(x, y):
    """ 
    Random Search for Hyperparameters classifiers
    """
    
    clsfs = [KNeighborsClassifier(), RandomForestClassifier(bootstrap=True, random_state=None), SVC(probability=True)]
    param_distributions = [{"leaf_size": randint(1,50), "n_neighbors": randint(1, 20), "p": [1,2]}, {"n_estimators": randint(1, 500),
                "max_features": randint(1, 30),
                "max_depth": randint(1, 20),
                "min_samples_leaf": randint(1, 20)}, {"C": randint(0.1, 100),
                "gamma": ['auto', 'scale'],
                "kernel": ['rbf', 'poly', 'sigmoid', 'linear']}]

    hyperparameters_clsfs = []
    for clf, param_dist in zip(clsfs, param_distributions):
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1)
        model = random_search.fit(x, y)
        parameters = model.best_estimator_.get_params()
        pprint(parameters)
        hyperparameters_clsfs.append(parameters)

    return hyperparameters_clsfs

hyperparameters = get_hyperparameters(x_train, y_train)

def cross_val_scores(x, y, hyperparameters, clf):
    '''
    Cross validation using a Logistic Regression classifier (5 folds)
    '''
    
    crss_val = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
    crss_val.get_n_splits(x, y)

    accuracies = []
    auc_scores = []
    specificities = []
    sensitivities = []
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

            predicted_probas = clf.predict_proba(x_val)[:, 1]
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
    plt.title(f'Receiver operating characteristic (ROC) curve {clf}')
    plt.grid()
    plt.show()

    return performance_scores

#%%

performance_clf = []
clsfs = [LogisticRegression(), KNeighborsClassifier(leaf_size=hyperparameters[0].get('leaf_size'), n_neighbors=hyperparameters[0].get('n_neighbors'), p=hyperparameters[0].get('p')), RandomForestClassifier(bootstrap=True, max_depth=hyperparameters[1].get('max_depth'), max_features=hyperparameters[1].get('max_features'), min_samples_leaf=hyperparameters[1].get('min_samples_leaf'), n_estimators=hyperparameters[1].get('n_estimators'), random_state=None), SVC(C=hyperparameters[2].get("C"), gamma=hyperparameters[2].get("gamma"), kernel=hyperparameters[2].get("kernel"), probability=True)]
clsfs_names =['Logistic Regression', 'kNN', 'Random Forest', 'SVM']

performance_clf_int = []
clsfs_int = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(bootstrap=True, random_state=None), SVC(probability=True)]
clsfs__int_names =['Logistic Regression', 'kNN', 'Random Forest', 'SVM']

for clf in clsfs:
    performances = cross_val_scores(x_train, y_train, hyperparameters, clf)
    performance_clf.append(performances)

for clf_int in clsfs_int:
    performances_int = cross_val_scores(x_train, y_train, hyperparameters, clf) 
    performance_clf_int.append(performances_int)

data1 = pd.DataFrame(performance_clf[0], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=1)
data2 = pd.DataFrame(performance_clf[1], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=2)
data3 = pd.DataFrame(performance_clf[2], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=3)
data4 = pd.DataFrame(performance_clf[3], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=4)

cdf = pd.concat([data1, data2, data3, data4])
mdf = pd.melt(cdf, id_vars=['Location'], var_name=['Index'])

ax = sns.boxplot(x="Location", y="value", hue="Index", data=mdf)    
plt.xticks([0, 1, 2, 3], ['Logistic Regression', 'kNN', 'Random Forest', 'SVM'])
ax.set_xlabel('Classifier')
ax.set_ylabel('Performance')
plt.show()

#%% Evaluate hyperparameters

def evaluate(model, x, y):
    predictions = model.predict(x)
    errors = abs(predictions - y)
    mape = 100 * np.mean(errors / y)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
    
base_model = RandomForestClassifier(bootstrap=True, random_state = 42)
base_model.fit(x_train, y_train)
base_accuracy = evaluate(base_model, x_train, y_train)

best_random = RandomForestClassifier(bootstrap=True, max_depth=hyperparameters[1].get('max_depth'), max_features=hyperparameters[1].get('max_features'), min_samples_leaf=hyperparameters[1].get('min_samples_leaf'), n_estimators=hyperparameters[1].get('n_estimators'), random_state=None)
best_random.fit(x_train, y_train)
random_accuracy = evaluate(best_random, x_train, y_train)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# Op internet fitten ze op de training set en bepalen ze de acuracy op de test set. 
# Moet dat bij ons dat train en val zijn?

# %%
