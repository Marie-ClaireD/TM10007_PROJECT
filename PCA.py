'''
Determine the number of principal components for the PCA
'''
#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut 
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model 
import matplotlib.pyplot as plt
import numpy as np

import statistics
#%%

from hn.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

#%%
filepath = '/Users/quintenmank/Desktop/TM10007/TM10007_PROJECT/TM10007_PROJECT-1/hn/HN_radiomicFeatures.csv'
data = np.genfromtxt(filepath, delimiter=',', dtype='float64')
scaler = MinMaxScaler(feature_range=[0,1])
data_rescaled = scaler.fit_transform(data[1:, 1:])
data_rescaled = np.nan_to_num(data_rescaled)

pca = PCA().fit(data_rescaled)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance (%)')
plt.show()

# %%
