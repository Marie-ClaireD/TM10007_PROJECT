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
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
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
import seaborn as sns
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
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, n_jobs=-1)
        model = random_search.fit(x, y)
        parameters = model.best_estimator_.get_params()
        pprint(parameters)
        hyperparameters_clsfs.append(parameters)

    return hyperparameters_clsfs

hyperparameters = get_hyperparameters(x_train, y_train)


#%%

def cross_val_scores_l1(x, y, hyperparameters, clf):
    '''
    Cross validation using a Logistic Regression classifier (5 folds)
    '''
    numerics = ['int16','int32','int64','float16','float32','float64']
    numerical_vars = list(data.select_dtypes(include=numerics).columns)
    data1 = data[numerical_vars]

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
        y_train, y_val= y[train_index], y[val_index]
        scaler = StandardScaler()
        scaler.fit(pd.DataFrame(x_train).fillna(0))
        lasso = SelectFromModel(estimator=Lasso(alpha=0.035, random_state=None, max_iter=10000), threshold='median')
        lasso.fit(scaler.transform(pd.DataFrame(x_train).fillna(0)), y_train) 
        lasso.get_support()
        selected_feat = data1.columns[(lasso.get_support())]        
        print('total features: {}'.format((x_train.shape[1])))
        print('features with coefficients shrank to zero: {}'.format(
     np.sum(lasso.estimator_.coef_ == 0)))
        # Getting a list of removed features
        removed_feats = [(lasso.estimator_.coef_ == 0).ravel().tolist()]
        # print(removed_feats)

        # Remove features from training and validation set
        x_train_selected = lasso.transform(pd.DataFrame(x_train).fillna(0))
      
        x_val_selected = lasso.transform(pd.DataFrame(x_val).fillna(0))
        

        clf.fit(x_train_selected, y_train)
        prediction = clf.predict(x_val_selected)

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

        predicted_probas = clf.predict_proba(x_val_selected)[:, 1]
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

performance_clf = []
clsfs = [LogisticRegression(), KNeighborsClassifier(n_neighbors=hyperparameters[0].get('n_neighbors')), RandomForestClassifier(bootstrap=True, max_depth=hyperparameters[1].get('max_depth'), max_features=hyperparameters[1].get('max_features'), min_samples_leaf=hyperparameters[1].get('min_samples_leaf'), n_estimators=hyperparameters[1].get('n_estimators'), random_state=None), SVC(C=hyperparameters[2].get("C"), gamma=hyperparameters[2].get("gamma"), kernel=hyperparameters[2].get("kernel"), probability=True)]
clsfs_names =['Logistic Regression', 'kNN', 'Random Forest', 'SVM']

for clf in clsfs:
    performances = cross_val_scores_l1(x_train, y_train, hyperparameters, clf) 
    performance_clf.append(performances)

for item in performance_clf: 
    plt.figure()
    item.boxplot()
    plt.show()

# %% NEW


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data1 = data[numerical_vars]
clsfs = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(bootstrap=True, random_state=None), SVC(probability=True, max_iter=10**7)]
names = ['Logistic Regression', 'kNN', 'Random Forest', 'SVM']
param_distributions = [{'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                        'max_iter': randint(1, 100)}, {'leaf_size': randint(1, 50),
                        'n_neighbors': randint(1, 20), 'p': [1, 2]}, {'n_estimators': randint(1, 500),
                        'max_features': randint(1, 30), 'max_depth': randint(1, 20),
                        'min_samples_leaf': randint(1, 20)},{'C': randint(0.1, 100),
                        'gamma': ['auto', 'scale'], 'kernel': ['rbf','sigmoid', 'linear', 'poly']}]
performance_clf = []
for clf,name, param_dist in zip(clsfs, names, param_distributions):
    accuracies = []
    auc_scores = []
    specificities = []
    sensitivities = []
    tprs = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)

    # performance_scores = pd.DataFrame()
    # models = []
    # accuracy = []
    crss_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=None) 
    for train_index, test_index in crss_val.split(features, labels):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test= labels[train_index], labels[test_index]

        scaler = StandardScaler()
        scaler.fit(pd.DataFrame(x_train).fillna(0))
        lasso = SelectFromModel(estimator=Lasso(alpha=0.035))
        lasso.fit(scaler.transform(pd.DataFrame(x_train).fillna(0)), y_train) 
        lasso.get_support()
        selected_feat = data1.columns[(lasso.get_support())]        
        print('total features: {}'.format((x_train.shape[1])))
        print('features with coefficients shrank to zero: {}'.format(np.sum(lasso.estimator_.coef_ == 0)))
        # Getting a list of removed features
        removed_feats = [(lasso.estimator_.coef_ == 0).ravel().tolist()]

        # Remove features from training and validation set
        x_train = lasso.transform(pd.DataFrame(x_train).fillna(0))
      
        x_test = lasso.transform(pd.DataFrame(x_test).fillna(0))
        

        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', n_jobs=-1)
        model = random_search.fit(x_train, y_train)
        model = model.best_estimator_
        #models.append(model)
        prediction = model.predict(x_test)

        performance_scores = pd.DataFrame()                  
        auc_scores.append(roc_auc_score(y_test, prediction))
        conf_mat = confusion_matrix(y_test, prediction)
        total = sum(sum(conf_mat))
        accuracies.append((conf_mat[0, 0]+conf_mat[1, 1])/total)
        sensitivities.append(conf_mat[0, 0]/(conf_mat[0, 0]+conf_mat[0, 1]))
        specificities.append(conf_mat[1, 1]/(conf_mat[1, 0]+conf_mat[1, 1]))
        performance_scores['Accuracy'] = accuracies
        performance_scores['AUC'] = auc_scores
        performance_scores['Sensitivity'] = sensitivities
        performance_scores['Specificity'] = specificities

        predicted_probas = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, predicted_probas)
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
    plt.title(f'Receiver operating characteristic (ROC) curve {name}')
    plt.grid()
    plt.show()

    performance_clf.append(performance_scores)

data1 = pd.DataFrame(performance_clf[0], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=1)
data2 = pd.DataFrame(performance_clf[1], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=2)
data3 = pd.DataFrame(performance_clf[2], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=3)
data4 = pd.DataFrame(performance_clf[3], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=4)

cdf = pd.concat([data1, data2, data3, data4])
mdf = pd.melt(cdf, id_vars=['Location'], var_name=['Index'])


ax = sns.boxplot(x="Location", y="value", hue="Index", data=mdf)    
plt.xticks([0, 1, 2, 3], names)
ax.set_xlabel('Classifier')
ax.set_ylabel('Performance')
plt.show()

    # performance_clf = []
    # clsfs = [LogisticRegression(), KNeighborsClassifier(leaf_size=hyperparameters[0].get('leaf_size'), n_neighbors=hyperparameters[0].get('n_neighbors'), p=hyperparameters[0].get('p')), RandomForestClassifier(bootstrap=True, max_depth=hyperparameters[1].get('max_depth'), max_features=hyperparameters[1].get('max_features'), min_samples_leaf=hyperparameters[1].get('min_samples_leaf'), n_estimators=hyperparameters[1].get('n_estimators'), random_state=None), SVC(C=hyperparameters[2].get("C"), gamma=hyperparameters[2].get("gamma"), kernel=hyperparameters[2].get("kernel"), probability=True)]
    # clsfs_names =['Logistic Regression', 'kNN', 'Random Forest', 'SVM']

    # for clf in clsfs:
    #     performances = cross_val_scores(x_train, y_train, hyperparameters, clf_int) 
    #     performance_clf.append(performances_int)

    # data1 = pd.DataFrame(performance_clf[0], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=1)
    # data2 = pd.DataFrame(performance_clf[1], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=2)
    # data3 = pd.DataFrame(performance_clf[2], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=3)
    # data4 = pd.DataFrame(performance_clf[3], columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity']).assign(Location=4)

    # cdf = pd.concat([data1, data2, data3, data4])
    # mdf = pd.melt(cdf, id_vars=['Location'], var_name=['Index'])


    # ax = sns.boxplot(x="Location", y="value", hue="Index", data=mdf)    
    # plt.xticks([0, 1, 2, 3], ['Logistic Regression', 'kNN', 'Random Forest', 'SVM'])
    # ax.set_xlabel('Classifier')
    # ax.set_ylabel('Performance')
    # plt.show()
    

# %%
