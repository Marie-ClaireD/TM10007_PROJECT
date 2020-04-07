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

#%%
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


    return performance_scores


performance_clf = []
clsfs = [LogisticRegression(), KNeighborsClassifier(n_neighbors=hyperparameters[0].get('n_neighbors')), RandomForestClassifier(bootstrap=True, max_depth=hyperparameters[1].get('max_depth'), max_features=hyperparameters[1].get('max_features'), min_samples_leaf=hyperparameters[1].get('min_samples_leaf'), n_estimators=hyperparameters[1].get('n_estimators'), random_state=None), SVC(C=hyperparameters[2].get("C"), gamma=hyperparameters[2].get("gamma"), kernel=hyperparameters[2].get("kernel"), probability=True)]
clsfs_names =['Logistic Regression', 'kNN', 'Random Forest', 'SVM']
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in clsfs:
    model = cls.fit(x_train, y_train)
    yproba = model.predict_proba(x_val)[::,1]
    
    fpr, tpr, _ = roc_curve(y_val,  yproba)
    auc = roc_auc_score(y_val, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


for clf in clsfs:
    performances = cross_val_scores(x_train, y_train, hyperparameters, clf) 
    performance_clf.append(performances)

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








# %%
