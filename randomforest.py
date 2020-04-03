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
from numpy import interp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from hn.load_data import load_data
from pprint import pprint

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
def random_forest(x, y):
    """ 
    Random Forest
    """
    crss_val = RepeatedKFold(n_splits = 5, n_repeats=10, random_state = None)           
    crss_val.get_n_splits(x, y)

    n_trees = [10, 20, 50, 100]
    predict_labels = []
    predict_probas = []
    y_val_total = []

    #idx = np.arange(0, len(y))
    
    for train_index, val_index in crss_val.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val= y[train_index], y[val_index]

        if min(x_train.shape[0], x_train.shape[1]) < 70:
            print('Not enough input values for PCA with 70 components')
            sys.exit()
        else:
            #pca = PCA(n_components=70)
            #pca.fit(x_train)
            #x_train = pca.transform(x_train)
            #x_val = pca.transform(x_val)
            for n_tree in n_trees:
                clf = RandomForestClassifier(n_estimators = n_tree, bootstrap=True, random_state=None)
                clf.fit(x, y) 
                prediction=clf.predict(x_val)
                predict_labels.append(prediction)
                predict = clf.predict_proba(x_val)[:,1]
                predict_probas.append(predict)
                y_val_total.append(y_val)

    predict_labels = np.array(predict_labels)
    predict_probas = np.array(predict_probas)
    print(predict_labels)
    print(predict_probas)

    return predict_labels, predict_probas, y_val_total

predict_labels_rf, predict_proba_rf, y_val_total_rf = random_forest(x_train, y_train)

#%% Random Forest n_estimator search
# Deze klopt nog niet ivm cross validation
def random_forest_search(x, y):
    """ 
    Random Forest Random Search for Hyperparameters
    """

    crss_val = RepeatedKFold(n_splits=5, n_repeats=1, random_state=None)           
    crss_val.get_n_splits(x, y)

#    predict_labels = []
#    predict_probas = []
#    y_val_total = []
    
    accuracy_rs = []
    accuracies = []
    auc_scores = []
    specificities = []
    sensitivities = []
    tprs = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)

    # param_dist={"n_estimators":[1, 5, 20, 100], "max_features":randint(5, 30),"max_depth":randint(2, 18),"min_samples_leaf":randint(2, 17)}
    # for x_train, y_train in split_sets(features, labels): ??
    for train_index, val_index in crss_val.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
#        if min(x_train.shape[0], x_train.shape[1]) < 70:
#            print('Not enough input values for PCA with 70 components')
#            sys.exit()

#       else:
#            pca = PCA(n_components=70)
#            pca.fit(x_train)
#            x_train = pca.transform(x_train)
#            x_val = pca.transform(x_val)

        param_dist = {"n_estimators": randint(1, 200),
                        "max_features": randint(5, 30),
                        "max_depth": randint(2, 18),
                        "min_samples_leaf": randint(1, 17)}
        clf = RandomForestClassifier(bootstrap=True, random_state=None)
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, n_jobs=-1) #Hier nog een keer CV?
        model = random_search.fit(x_train, y_train)
#        prediction = random_search.predict(x_val)
#        predict_labels.append(prediction)
#        predict = random_search.predict_proba(x_val)[:,1]
#        predict_probas.append(predict)
#        y_val_total.append(y_val)
    
#    predict_labels = np.array(predict_labels)
#    predict_probas = np.array(predict_probas)
    #print(predict_labels)
    #print(predict_probas)
        
        prediction = random_search.predict(x_val)        
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

    pprint(model.best_estimator_.get_params())

    return performance_scores

performances = random_forest_search(x_train, y_train)
print(performances)

    
#    return predict_labels, predict_probas, y_val_total

# predict_labels_rf_s, predict_proba_rf_s, y_val_total_rf_s = random_forest_search(x_train, y_train)
#%%
#%% Random Forest n_estimator search GOEDE
def random_forest_search(x, y):
    """ 
    Random Forest Random Search for Hyperparameters
    """

    crss_val = RepeatedKFold(n_splits=5, n_repeats=1, random_state=None)           
    crss_val.get_n_splits(x, y)

#    predict_labels = []
#    predict_probas = []
#    y_val_total = []
    
    accuracy_rs = []
    accuracies = []
    auc_scores = []
    specificities = []
    sensitivities = []
    tprs = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)

    param_dist = {"n_estimators": randint(1, 200),
                    "max_features": randint(5, 30),
                    "max_depth": randint(2, 18),
                    "min_samples_leaf": randint(1, 17)}
    clf = RandomForestClassifier(bootstrap=True, random_state=None)
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, n_jobs=-1) #Hier nog een keer CV?
    model = random_search.fit(x, y)
    hyperparameters = model.best_estimator_.get_params()
    pprint(hyperparameters)

    for train_index, val_index in crss_val.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        clf = RandomForestClassifier(bootstrap=True, max_depth=hyperparameters.get('max_depth'), max_features=hyperparameters.get('max_features'), min_samples_leaf=hyperparameters.get('min_samples_leaf'), n_estimators=hyperparameters.get('n_estimators'), random_state=None)
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

performances = random_forest_search(x_train, y_train)
print(performances)
#%%
  # Hyperparameters: 
    # n_estimators: number of decision trees
    # bootstrap = True 
    # max_depth: default is none, so the decision trees can be prone to overfitting. --> other value has to be given. 
    # min_samples_leaf: specifies the minimum number of samples required to be at a leaf node. The default value for this parameter is 1, which means that every leaf      must have at least 1 sample that it classifies.
    # random_state = 0: to obtain a deterministic behaviour during fitting

    # Vb. Exhaustive Grid search:
    # n_estimators = [100, 300, 500, 800, 1200]
    # max_depth = [5, 8, 15, 25, 30]
    # min_samples_split = [2, 5, 10, 15, 100]
    # min_samples_leaf = [1, 2, 5, 10] 

    # hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
                  # min_samples_split = min_samples_split, 
                  # min_samples_leaf = min_samples_leaf)

    # gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, n_jobs = -1)
    # bestF = gridF.fit(x_train, y_train)


