"""
ref_machine_learning
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from pandas.plotting import scatter_matrix
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


mass = pd.read_table('feature_ref.txt', sep='\t',delim_whitespace=True,index_col=False, names=['filename','classe','area',' perimeter', 	 'circularity', 	' mu_NRL' ,	 'std_NRL', 	' zero_crossing', 	 'max_axis', 	' min_axis', 	'mu_VR', 	 'std_VR', 	 'RLE' ,	'convexity' ,	 'mu_I' 	' std_I', 'kurtosis', 	 'skewness'])


print(mass.shape)  #'abbiamo 59 elementi e 7 feature'

print('-----------------------------------')

print(mass['classe'].unique())
print(mass.groupby('classe').size())

print('-----------------------------------')

sns.countplot(mass['classe'],label="Count")
plt.grid()
plt.show()


'''
Box plot for each numeric variable will give us a clearer
idea of the distribution of the input variables:
'''

plt.figure()
mass.drop('area', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), title='Box Plot for each input variable')
plt.savefig('mass_box')
plt.show()

plt.figure()
fruits.drop('classe' ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('mass_hist')
plt.show()


#plt.figure()
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('mass_scatter_matrix')
#plt.show()

