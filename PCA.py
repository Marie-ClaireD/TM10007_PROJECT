'''
Determine the number of principal components for the PCA
'''
# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


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
