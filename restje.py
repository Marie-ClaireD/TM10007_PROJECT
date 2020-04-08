#%%
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix, roc_auc_score
from scipy import interp
from scipy.stats import randint
from pprint import pprint

# The training set is again divided into a training and a validation set and afterwards classified using a leave one out validation and logistic regression.


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

# Features from data
features = data.loc[:, data.columns != 'label'].values

# Labels from data
labels = data.loc[:,['label']].values
#low risk patients receive the label 0 and high risk the label 1
labels = [item if item!='T12' else 0 for item in labels]
labels = [item if item!='T34' else 1 for item in labels]
labels = np.array(labels)
#number of high and low risk patients is printed to the terminal
print(f'Number of high risk patients: {np.count_nonzero(labels)}') 
print(f'Number of low risk patients: {len(labels) - np.count_nonzero(labels)}')

#%% 

def split_sets(features, labels):
    """
    Splits the features and labels into a training set (80%) and test set (20%)
    """
    x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=None)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test 

x_train, x_test, y_train, y_test = split_sets(features, labels) 

#%%

crss_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=None) 

for train_index, test_index in crss_val.split(features, labels):
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    pca = PCA(n_components=47)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    clsfs = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(bootstrap=True, random_state=None), SVC(probability=True)]
    param_distributions = [{"penalty": ['l1', 'l2', 'elasticnet', 'none'],
                            "max_iter": randint(1, 200)}, {"leaf_size": randint(1,50), 
                            "n_neighbors": randint(1, 20), "p": [1,2]}, {"n_estimators": randint(1, 500),
                            "max_features": randint(1, 30), "max_depth": randint(1, 20),
                            "min_samples_leaf": randint(1, 20)}, {"C": randint(0.1, 100),
                            "gamma": ['auto', 'scale'], "kernel": ['rbf', 'poly', 'sigmoid', 'linear']}]

    hyperparameters_clsfs = []
    
    for clf, param_dist in zip(clsfs, param_distributions):
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, n_jobs=-1)
        model = random_search.fit(x_train, y_train)
        parameters = model.best_estimator_.get_params()
        #pprint(parameters)
        hyperparameters_clsfs.append(parameters)
        values = pd.DataFrame(model.cv_results_)
    
    hyperparameters = hyperparameters_clsfs
    print(hyperparameters)

    clsfs = [LogisticRegression(penalty=hyperparameters[0].get('penalty'), max_iter=hyperparameters[0].get('max_iter')), KNeighborsClassifier(leaf_size=hyperparameters[1].get('leaf_size'), n_neighbors=hyperparameters[1].get('n_neighbors'), p=hyperparameters[1].get('p')), RandomForestClassifier(bootstrap=True, max_depth=hyperparameters[2].get('max_depth'), max_features=hyperparameters[2].get('max_features'), min_samples_leaf=hyperparameters[2].get('min_samples_leaf'), n_estimators=hyperparameters[2].get('n_estimators'), random_state=None), SVC(C=hyperparameters[3].get("C"), gamma=hyperparameters[3].get("gamma"), kernel=hyperparameters[3].get("kernel"), probability=True)]
    clsfs_names =['Logistic Regression', 'kNN', 'Random Forest', 'SVM']

#%%
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










def cross_val(x,y):
    """
    Cross validation using a Logistic Regression classifier (5 folds)
    """

    crss_val = RepeatedKFold(n_splits = 5, n_repeats=10, random_state = None)           
    crss_val.get_n_splits(x, y)

    predict_labels = []
    predict_probas = []
    y_val_total = []


    for train_index, val_index in crss_val.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val= y[train_index], y[val_index]

        if min(x_train.shape[0], x_train.shape[1]) < 70:
            print('Not enough input values for PCA with 70 components')
            sys.exit()
        else:
            pca = PCA(n_components=70)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_val = pca.transform(x_val)

            lrg=LogisticRegression(solver='lbfgs')
            lrg.fit(x_train,y_train) 
            prediction=lrg.predict(x_val)
            predict_labels.append(prediction)
            predict = lrg.predict_proba(x_val)[:,1]
            predict_probas.append(predict)


    predict_labels = np.array(predict_labels)
    predict_probas = np.array(predict_probas)
    return predict_labels, predict_probas, y_val_total

predict_labels_crss, predict_proba_crss, y_val_total_crss = cross_val(x_train, y_train)





def compute_performance(labels_test, predict_test):
    """
    Compute accuracy, sensitivity and specificity based on confusion matrix.
    """
    try:
        # Form confusion matrix
        conf_matrix = confusion_matrix(labels_test, predict_test).ravel()

        # Calculate accuracy
        accuracy = (conf_matrix[0]+conf_matrix[3])/(conf_matrix[0] +
                                                    conf_matrix[1] + conf_matrix[2]
                                                    + conf_matrix[3])
        # Calculate sensitivity
        sensitivity = conf_matrix[3]/(conf_matrix[3]+conf_matrix[2])

        # Calculate specificity
        specificity = conf_matrix[0]/(conf_matrix[0]
                                      +conf_matrix[1])

    except Exception as error:
        print("Unexpected error:"+str(error))
        raise
    
    return(accuracy, sensitivity, specificity)

