#%%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold 
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix, roc_auc_score
from scipy import interp
from scipy.stats import randint
from pprint import pprint
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
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

# Extract labels from data
labels = data.loc[:,['label']].values

# Low risk patients receive the label 0 and high risk the label 1
labels = [item if item!='T12' else 0 for item in labels]
labels = [item if item!='T34' else 1 for item in labels]
labels = np.array(labels)

# Number of high and low risk patients is printed
print(f'Number of high risk patients: {np.count_nonzero(labels)}')
print(f'Number of low risk patients: {len(labels) - np.count_nonzero(labels)}')

# Drop features that are not distinctive (standard deviation equals 0)
data_features = data.drop(['label'], axis=1)
columns = data_features.columns
std = data_features[columns].std()
not_distinctive = std[std==0].index
data_features = data_features.drop(not_distinctive, axis=1)
print(f'\nFeatures that were considered non-distinctive : {not_distinctive}')

# Extract features from data
features = data_features.values
print(f'Number of samples: {features.shape[0]}')
print(f'Resulting number of features: {features.shape[1]}')
#%%

def split_sets(x, y):
    """
    Splits the features and labels into a training set (80%) and test set (20%)
    """
    #use repeated stratified KFold
    crss_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    
    #splitting data into test and training set
    for train_index, test_index in crss_val.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #scale the data by fitting on the training set and transforming the test set
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        return x_train, x_test, y_train, y_test    

x_train, x_test, y_train, y_test = split_sets(features, labels)

def scale_data(x, y):
    """Scale data with Standard scaler"""

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)
    x_test = scaler.transform(y)
    return x_train, x_test

# Tuning alpha parameter for optimal feature selection with LASSO
test_scores = []
a_max=0
n_alphas = 100
alphas = np.logspace(-2,0,num=n_alphas)

# Loop over range of alphas
for a in alphas:
    # Fit classifier
    clf = Lasso(alpha=a)
    clf.fit(pd.DataFrame(x_train).fillna(0), y_train)
        
    # Compute determination coefficient and selecting best alpha
    test_score = clf.score(x_test, y_test)
    if test_score > a_max:
         a_max = test_score
         best_alpha = a
 
# The value used for alpha = 0.040  

def apply_lasso(x1, y, x2, data):
    """
    Apply L1 regularization with best alpha
    """
    # Apply LASSO with calculated best alpha
    lasso = SelectFromModel(estimator=Lasso(alpha=0.040))
    lasso.fit((pd.DataFrame(x1).fillna(0)),y)
    lasso.get_support()

    # Get list of removed features and remove from training and test set
    x_train = lasso.transform(pd.DataFrame(x1).fillna(0))
    x_test = lasso.transform(pd.DataFrame(x2).fillna(0))
    return x_train, x_test

x_train_L1, x_test_L1 = apply_lasso(x_train, y_train, x_test, data_features)
print('Resulting Number of Features: {}'.format((x_train_L1.shape[1])))


def apply_pca(x, y):
    """Apply PCA with 47 components to data"""

    pca = PCA(n_components=47)
    pca.fit(x)
    x_train = pca.transform(x)
    x_test = pca.transform(y)
    return x_train, x_test

def compute_performance(model, x, y):
    """ Get Performances on test set"""

    base_fpr = np.linspace(0, 1, 101)

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

    return performance_scores, tprs, aucs

def plot_ROC(tprs, aucs, name):
    base_fpr = np.linspace(0, 1, 101)

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

    return

def create_boxplot(performance_clf, names):
    
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
    
    return

def evaluate_hyperparameters(x, y):
    """Evaluate Hyperparameters"""
    
    fit_models, models = get_hyperparameters(x, y)
    for fit_model, model in zip(fit_models, models):
        parameters = fit_model.best_estimator_.get_params()
        best_score = fit_model.best_score_
        print(f'Best score classifier = {best_score}')
        print(f'For model with hyperparameters: {model}')

#%% compare L1 with PCA
# Always good to set a seed for reproducibility

from pandas import set_option

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from statistics import mean

def GetBasedModel():
    basedModels = []
    basedModels.append((LogisticRegression()))
    basedModels.append((KNeighborsClassifier()))
    basedModels.append((RandomForestClassifier(bootstrap=True, random_state=None))) 
    basedModels.append((SVC(probability=True)))
    return basedModels


def GetBasedModelHyper(model_clf):
    basedModelsHyper = []
    for model in model_clf:
        basedModelsHyper.append(model)

    return basedModelsHyper


def BasedLine2(X_train, y_train, models):
    # Test options and evaluation metric
    results = []
 
    for model in models:
        kfold = RepeatedStratifiedKFold(n_splits=5, random_state=None)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        # msg = " %f (%f)" % (cv_results.mean(), cv_results.std())
        # print(msg)
        
    return results

def ScoreDataFrame(results):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
    
        return float(prc.format(f_val))

    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))

    scoreDataFrame = pd.DataFrame({'Accuracy score': scores})
    return scoreDataFrame

#%% Assignment without hyperparameters
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold 
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix, roc_auc_score
from scipy import interp
from scipy.stats import randint
from pprint import pprint
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import seaborn as sns
clsfs = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(bootstrap=True, random_state=None), SVC(probability=True)]
names = ['Logistic Regression', 'kNN', 'Random Forest', 'SVM']
#param_distributions = [{}, {}, {}, {}]

baseline_accuracy = []
baseline_PCA = []
baseline_L1 = []
for clf, name, param_dist in zip(clsfs, names, param_distributions):
    crss_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=None) 
    results_rs = []
    results_PCA = []
    results_L1 = []
    for train_index, test_index in crss_val.split(features, labels):

        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Scale data with Standard Scalar
        x_train, x_test = scale_data(x_train, x_test)

        
        # Apply PCA to data
        x1_train, x1_test = apply_pca(x_train, x_test)
 


        # Apply L1 to data
        x2_train, x2_test = apply_lasso(x_train, y_train, x_test, data_features)

        #RandomizedSearch for optimalization Hyperparameters
#        random_search1 = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', n_jobs=-1)
#        model_randomsearch_base = random_search1.fit(x_train, y_train)
#        random_search2 = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', n_jobs=-1)
#        model_randomsearch_PCA = random_search2.fit(x1_train, y_train)
#        random_search3 = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', n_jobs=-1)
#        model_randomsearch_L1 = random_search3.fit(x2_train, y_train)
#
#        # Results Randomized Search
#        result_rs = model_randomsearch_base.best_score_
#        results_rs.append(result_rs)
#        result_PCA = model_randomsearch_PCA.best_score_
#        results_PCA.append(result_PCA)
#        result_L1 = model_randomsearch_L1.best_score_
#        results_L1.append(result_L1)
#        
        model_base = clf.fit(x_train, y_train)
        results_base = cross_val_score(model_base, x_train, y_train, cv=5)
        model_PCA = clf.fit(x1_train, y_train)
        results_PCA = cross_val_score(model_PCA, x1_train, y_train, cv=5)
        model_L1 = clf.fit(x2_train, y_train)
        results_L1 = cross_val_score(model_L1, x2_train, y_train, cv=5)

    # Mean accuracies
    results_base = mean(results_base)
    baseline_accuracy.append(results_base)
    results_PCA = mean(results_PCA)
    baseline_PCA.append(results_PCA)
    results_L1 = mean(results_L1)
    baseline_L1.append(results_L1)
# Create dataframe and append results 
base_model = pd.DataFrame()
base_model['Classifier'] = ['LR', 'KNN', 'RF', 'SVM']
base_model['Baseline Accuracy'] = baseline_accuracy
base_PCA = pd.DataFrame()
base_PCA['PCA Accuracy'] = baseline_PCA
base_L1 = pd.DataFrame()
base_L1['L1 Accuracy'] = baseline_L1


# With Hyperparameters
clsfs = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(bootstrap=True, random_state=None), SVC(probability=True)]
names = ['Logistic Regression', 'kNN', 'Random Forest', 'SVM']
param_distributions = [{}, {'n_neighbors': randint(1, 20)}, {'n_estimators': randint(1, 500),
                        'max_features': randint(1, 30), 'max_depth': randint(1, 20),
                        'min_samples_leaf': randint(1, 20)}, {'C': randint(0.1, 100),
                        'gamma': ['auto', 'scale'], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}]
HP_accuracy = []
HP_PCA = []
HP_L1 = []
for clf, name, param_dist in zip(clsfs, names, param_distributions):
    crss_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=None) 
    results_rs = []
    results_PCA = []
    results_L1 = []
    for train_index, test_index in crss_val.split(features, labels):

        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Scale data with Standard Scalar
        x_train, x_test = scale_data(x_train, x_test)

        
        # Apply PCA to data
        x1_train, x1_test = apply_pca(x_train, x_test)


        # Apply L1 to data
        x2_train, x2_test = apply_lasso(x_train, y_train, x_test, data_features)

        #RandomizedSearch for optimalization Hyperparameters
        random_search1 = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', n_jobs=-1)
        model_randomsearch_base = random_search1.fit(x_train, y_train)
        random_search2 = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', n_jobs=-1)
        model_randomsearch_PCA = random_search2.fit(x1_train, y_train)
        random_search3 = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', n_jobs=-1)
        model_randomsearch_L1 = random_search3.fit(x2_train, y_train)
        
        # Results 
        result_rs = model_randomsearch_base.best_score_
        results_rs.append(result_rs)
        result_PCA = model_randomsearch_PCA.best_score_
        results_PCA.append(result_PCA)
        result_L1 = model_randomsearch_L1.best_score_
        results_L1.append(result_L1)


    results_rs = mean(results_rs)
    HP_accuracy.append(results_rs)
    results_PCA = mean(results_PCA)
    HP_PCA.append(results_PCA)
    results_L1 = mean(results_L1)
    HP_L1.append(results_L1)
HP_acc = pd.DataFrame()
HP_acc['Accuracy HP'] = HP_accuracy

HP_PC = pd.DataFrame()
HP_PC['Accuracy PCA HP'] = HP_PCA

HP_L = pd.DataFrame()
HP_L['Accuracy L1 HP'] = HP_L1


compareModels = pd.concat([base_model, HP_acc], axis=1)
compareModels['Improvement Base %']= (100 *(compareModels['Accuracy HP'] - compareModels['Baseline Accuracy'])/compareModels['Baseline Accuracy'])
compareModels = pd.concat([compareModels, base_PCA, HP_PC], axis=1)
compareModels['Improvement PCA %']= (100 *(compareModels['Accuracy PCA HP']- compareModels['PCA Accuracy'])/compareModels['PCA Accuracy'])
compareModels = pd.concat([compareModels,base_L1, HP_L], axis=1)
compareModels['Improvement L1 %']= (100 *(compareModels['Accuracy L1 HP']- compareModels['L1 Accuracy'])/compareModels['L1 Accuracy'])


pprint(compareModels.round(2))

#%%
clsfs = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(bootstrap=True, random_state=None), SVC(probability=True)]
names = ['Logistic Regression', 'kNN', 'Random Forest', 'SVM']
param_distributions = [{}, {'n_neighbors': randint(1, 20)}, {'n_estimators': randint(1, 500),
                        'max_features': randint(1, 17), 'max_depth': randint(1, 20),
                        'min_samples_leaf': randint(1, 20)}, {'C': randint(0.1, 100),
                        'gamma': ['auto', 'scale'], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}]

performance_clf = []
performance = []
for clf, name, param_dist in zip(clsfs, names, param_distributions):
    accuracies = []
    auc_scores = []
    specificities = []
    sensitivities = []
    tprs = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)
    crss_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=None) 
    performance_scores = []
    
    for train_index, test_index in crss_val.split(features, labels):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Scale data with Standard Scalar
        x_train, x_test = scale_data(x_train, x_test)

        # Apply LASSO to data
        x_train, x_test = apply_lasso(x_train,y_train, x_test, data_features)

        # RandomSearch for optimalization Hyperparameters
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
        model = random_search.fit(x_train, y_train)
        model = model.best_estimator_

        # Evaluate performance Hyperparameters on test set
        performance_scores, tprs, aucs = compute_performance(model, x_test, y_test)

    performance_scores.loc['mean'] = performance_scores.mean()
    
    print(f'Performance {name} Classifier:')
    print(f'{performance_scores.tail(1)}')

    # Plot ROC curves
    plot_ROC(tprs, aucs, name)

    performance_clf.append(performance_scores)
    
#    perf = performance_scores.tail(1).values
#    performance.append(list(perf[0]))
#performance_tails = pd.DataFrame()
#performance_tails.append(performance)
#print(performance_tails)
#    
#    performance_test = pd.DataFrame()
#    performance_test['Classifier'] = ['LR', 'KNN', 'RF', 'SVM']
    #print(performance_test)
#    performance_result = pd.merge(performance_test, performance_scores.tail(1), how='inner', on = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity']
    
#    print(performance_result)

# Create boxplot
create_boxplot(performance_clf, names)