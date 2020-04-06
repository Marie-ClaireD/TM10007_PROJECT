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
                  "dual": ['False','True'],
                  "max_iter": randint(1, 200)}
clf = SVC(probability=True) 
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1) #Hier nog een keer CV?
model = random_search.fit(x_train, y_train)
hp_lg = model.best_estimator_.get_params()
pprint(hp_lg)
X = features
pca = PCA(n_components=70)
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
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
estimator = SVC()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0, 1.5),
                    cv=cv, n_jobs=4)

# SVM with hyperparameters
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
estimator = SVC(C=hyperparameters.get("C"), gamma=hyperparameters.get("gamma"), kernel=hyperparameters.get("kernel"), probability=True)
plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0, 1.5),
                    cv=cv, n_jobs=4)

from sklearn.ensemble import RandomForestClassifier
# Random Forest 
#title = "Learning Curves (Random Forest)"
#cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
#estimator = RandomForestClassifier()
#plot_learning_curve(estimator, title, X, y, axes=axes[:, 2], ylim=(0, 1.5),
#                    cv=cv, n_jobs=4)

# Random Forest with hyperparameters
#title = "Learning Curves (Random Forest)"
#cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
#estimator = RandomForestClassifier(bootstrap=True, max_depth=hyperparameters('max_depth'), max_features=hyperparameters('max_features'), min_samples_leaf=hyperparameters('min_samples_leaf'), n_estimators=hyperparameters('n_estimators'), random_state=None)
#plot_learning_curve(estimator, title, X, y, axes=axes[:, 3], ylim=(0, 1.5),
#                    cv=cv, n_jobs=4)
#from sklearn.linear_model import LogisticRegression

# Logistic regression 
#title = "Learning Curves (Logistic Regression)"
#cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
#estimator = LogisticRegression()
#plot_learning_curve(estimator, title, X, y, axes=axes[:, 4], ylim=(0, 1.5),
#                    cv=cv, n_jobs=4)

# Logistic regression with hyperparameters
#title = "Learning Curves (Logistic Regression)"
#cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
#estimator = LogisticRegression()
#plot_learning_curve(estimator, title, X, y, axes=axes[:, 5], ylim=(0, 1.5),
#                    cv=cv, n_jobs=4)
#plt.show()

# %%
# learning curve

plt.show()

clf = LogisticRegression()
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(clf, title,x_train, y_train, ylim=(0.3, 1.01), cv=cv)


# %%
# L1 Regularization

from sklearn.linear_model import Lasso
Xs = features
y = labels
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(Xs, y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

# %%
from sklearn.metrics import mean_squared_error, r2_score
model_lasso = Lasso(alpha=1)
model_lasso.fit(x_train, y_train)
pred_train_lasso = model_lasso.predict(x_train)
print(np.sqrt(mean_squared_error(y_train, pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))

# %%

# %%
 # L1 regularization

 import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
std = StandardScaler()
X_train_std = std.fit_transform(x_train)
X_test_std = std.fit_transform(x_test)
X_val_std = std.fit_transform(x_val)


# stores the weights of each feature
# does the same thing above except for lasso
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 2]
print('different alpha values:', alphas)

lasso_weight = []
for alpha in alphas:    
    lasso = Lasso(alpha = alpha, fit_intercept = True)
    lasso.fit(X_train_std, y_train)
    lasso_weight.append(lasso.coef_)


# %%
def weight_versus_alpha_plot(weight, alphas, features):
    """
    Pass in the estimated weight, the alpha value and the names
    for the features and plot the model's estimated coefficient weight 
    for different alpha values
    """
    fig = plt.figure(figsize = (8, 6))
    
    # ensure that the weight is an array
    weight = np.array(weight)
    for col in range(0,112, 1):
        plt.plot(alphas, weight[:, col], label = features[col])
        
    plt.axhline(0, color = 'black', linestyle = '--', linewidth = 3)
    
    # manually specify the coordinate of the legend
    plt.title('Coefficient Weight as Alpha Grows')
    plt.ylabel('Coefficient weight')
    plt.xlabel('alpha')
    
    return fig
# change default figure and font size
plt.rcParams['figure.figsize'] = 8, 6 
plt.rcParams['font.size'] = 12


lasso_fig = weight_versus_alpha_plot(lasso_weight, alphas, features)
# %%

# difference of lasso and ridge regression is that some of the coefficients can be zero i.e. some of the features are 
# completely neglected
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression



#print cancer_df.head(3)
X = features
Y = labels

lasso = Lasso()
lasso.fit(x_train,y_train)
train_score=lasso.score(x_train,y_train)
test_score=lasso.score(x_val, y_val)
coeff_used = np.sum(lasso.coef_!=0)
print("training score:", train_score)
print("test score: ", test_score)
print("number of features used: ", coeff_used)
lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(x_train,y_train)
train_score001=lasso001.score(x_train,y_train)
test_score001=lasso001.score(x_val, y_val)
coeff_used001 = np.sum(lasso001.coef_!=0)
print("training score for alpha=0.01:", train_score001) 
print("test score for alpha =0.01: ", test_score001)
print("number of features used: for alpha =0.01:", coeff_used001)
lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(x_train,y_train)
train_score00001=lasso00001.score(x_train,y_train)
test_score00001=lasso00001.score(x_val, y_val)
coeff_used00001 = np.sum(lasso00001.coef_!=0)
print("training score for alpha=0.0001:", train_score00001)
print("test score for alpha =0.0001: ", test_score00001)
print("number of features used: for alpha =0.0001:", coeff_used00001)
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_train_score=lr.score(x_train,y_train)
lr_test_score=lr.score(x_val, y_val)
print("LR training score:", lr_train_score) 
print("LR test score: ", lr_test_score)
plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.subplot(1,2,2)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.show()


# %%
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
reg = Lasso(alpha=1)
reg.fit(x_train, y_train)

print('Lasso Regression: R^2 score on training set', reg.score(x_train, y_train)*100)
print('Lasso Regression: R^2 score on test set', reg.score(x_val, y_val)*100)

# %%
