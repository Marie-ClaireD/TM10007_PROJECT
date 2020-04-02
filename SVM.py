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
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix
from scipy import interp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

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
    x_train, x_tmp, y_train, y_tmp = train_test_split(
    features, labels, train_size=0.8, random_state=1)

    x_val, x_test,y_val, y_test = train_test_split(
        x_tmp, y_tmp, train_size=0.5, random_state=1
    )
    return x_train, x_val, x_test, y_train, y_val, y_test 

x_train,x_val, x_test, y_train, y_val, y_test = split_sets(features, labels) 

#%%
def support_vector(x,y):
    """ 
    Support Vectorm Machine using Logistic Regression as a classifier
    """
    clf = SVC(kernel='linear', gamma='scale')   
    clf.fit(x, y)   
    x_train, x_val, x_test, y_train, y_val, y_test = split_sets(features, labels)

    predict_labels = []
    predict_probas = []
    y_val_total = []
    
    if min(x_train.shape[0], x_train.shape[1]) < 70:
            print('Not enough input values for PCA with 70 components')
            sys.exit()
    else:
        pca = PCA(n_components=70)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_val = pca.transform(x_val)

        lrg=LogisticRegression()
        lrg.fit(x_train,y_train) 
        prediction=lrg.predict(x_val)
        predict_labels.append(prediction)
        predict = lrg.predict_proba(x_val)[:,1]
        predict_probas.append(predict)
        y_val_total.append(y_val)

    predict_labels = np.array(predict_labels)
    predict_probas = np.array(predict_probas)
    print(predict_labels)
    #print(predict_probas)

    return predict_labels, predict_probas, y_val_total

predict_labels_svm, predict_proba_svm, y_val_total_svm = support_vector(x_train, y_train)
