'''
Determine the number of principal components for the PCA
'''
# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix
from scipy import interp
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

from hn.load_data import load_data
# %%

from hn.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

# %%
filepath = '/Users/quintenmank/Desktop/TM10007/TM10007_PROJECT/TM10007_PROJECT-1/hn/HN_radiomicFeatures.csv'
data = np.genfromtxt(filepath, delimiter=',', dtype='float64')
scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(data[1:, 1:])
data_rescaled = np.nan_to_num(data_rescaled)

pca = PCA().fit(data_rescaled)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance (%)')
plt.show()

# %%
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
# Learning curve

features = data.loc[:, data.columns != 'label'].values
features = StandardScaler().fit_transform(features)

# Labels
labels = data.loc[:,['label']].values
labels = [item if item!='T12' else 0 for item in labels]
labels = [item if item!='T34' else 1 for item in labels]
labels = np.array(labels)
print(f'Number of high risk patients: {np.count_nonzero(labels)}') 
print(f'Number of low risk patients: {len(labels) - np.count_nonzero(labels)}')
# %%
# learning curve
plt.show()
clf = LogisticRegression()
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(clf, x_train, y_train, cv=cv)
#%%

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def split_sets(x, y):
    '''
    Splits the features and labels into a training set (80%) and test set (20%)
    '''
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=None)
    return x_train, x_test, y_train, y_test
    

x_train, x_test, y_train, y_test = split_sets(features, labels)

def split_sets2(x,y):
    '''
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=None) 
    return x_train, x_val, y_train, y_val

x_train, x_val, y_train, y_val = split_sets2(x_train, y_train)


# hyperparameters random forest
param_dist = {"n_estimators": randint(1, 200),
                    "max_features": randint(5, 30),
                    "max_depth": randint(2, 18),
                    "min_samples_leaf": randint(1, 17)}
clf = RandomForestClassifier(bootstrap=True, random_state=None)
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1) #Hier nog een keer CV?
model = random_search.fit(x_train, y_train)
hp_rf = model.best_estimator_.get_params()
pprint(hp_rf)

#hyperparameters svm
param_dist = {"C": randint(0.1, 100),
                  "gamma": ['auto','scale'],
                  "kernel": ['rbf','poly','sigmoid','linear']}
clf = SVC(probability=True) 
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1) #Hier nog een keer CV?
model = random_search.fit(x_train, y_train)
hp_svm = model.best_estimator_.get_params()
pprint(hp_svm)

# hyperparameters logistic regression
param_dist = {"penalty": ['l1', 'l2', 'elasticnet', 'none'],
                  "max_iter": randint(1, 200)}
clf = LogisticRegression() 
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1) #Hier nog een keer CV?
model = random_search.fit(x_train, y_train)
hp_lg = model.best_estimator_.get_params()
pprint(hp_lg)
X = features
pca = PCA(n_components=30)
X = pca.fit_transform(X)
y = labels
#%%
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(0.1,1,5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.

        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 6, figsize=(10, 15))
title = "Learning Curves (Naive Bayes)"

# Gaussian Naive Bayes
#cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

#estimator = GaussianNB()
#plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.2, 1.5),
#                    cv=cv, n_jobs=4)

#title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVM
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
estimator = SVC()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0, 1.5),
                    cv=cv, n_jobs=4)

# SVM with hyperparameters
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
estimator = SVC(C=hp_svm.get("C"), gamma=hp_svm.get("gamma"), kernel=hp_svm.get("kernel"), probability=True)
plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0, 1.5),
                    cv=cv, n_jobs=4)

from sklearn.ensemble import RandomForestClassifier
# Random Forest 
title = "Learning Curves (Random Forest)"
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
estimator = RandomForestClassifier()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 2], ylim=(0, 1.5),
                    cv=cv, n_jobs=4)

# Random Forest with hyperparameters
title = "Learning Curves (Random Forest)"
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
estimator = RandomForestClassifier(bootstrap=True, max_depth=hp_rf.get('max_depth'), max_features=hp_rf.get('max_features'), min_samples_leaf=hp_rf.get('min_samples_leaf'), n_estimators=hp_rf.get('n_estimators'), random_state=None)
plot_learning_curve(estimator, title, X, y, axes=axes[:, 3], ylim=(0, 1.5),
                    cv=cv, n_jobs=4)

from sklearn.linear_model import LogisticRegression

# Logistic regression 
title = "Learning Curves (Logistic Regression)"
cv = RepeatedKFold(n_splits=2, n_repeats=10, random_state=None)
estimator = LogisticRegression()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 4], ylim=(0, 1.5),
                    cv=cv, n_jobs=4)

# Logistic regression with hyperparameters
title = "Learning Curves (Logistic Regression)"
cv = RepeatedKFold(n_splits=2, n_repeats=10, random_state=None)
estimator = LogisticRegression(penalty=hp_lg.get('penalty'), max_iter=hp_lg.get('max_iter'))
plot_learning_curve(estimator, title, X, y, axes=axes[:, 5], ylim=(0, 1.5),
                    cv=cv, n_jobs=4)
plt.show()

