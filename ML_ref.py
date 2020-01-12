"""
ref_machine_learning
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
