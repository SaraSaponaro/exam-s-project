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
mass.drop('classe', axis=1).plot(kind='box', subplots=True,title='Box Plot for each input variable') 
#plt.savefig('mass_box')


plt.figure()
mass.hist(bins='auto')
pl.suptitle("Histogram for each numeric input variable")
#plt.savefig('mass_hist')


'''
#plt.figure()
feature_names = ['area',' perimeter', 	 'circularity', 	' mu_NRL']
X = mass[feature_names]
y = mass['classe']
cmap = cm.get_cmap('cividis')
scatter = scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':25}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
#plt.savefig('mass_scatter_matrix')
'''

plt.show()


#%%
'''
We can see that the numerical values do not have the same scale.
We will need to apply scaling to the test set that we computed for the training set.
feature_names = ['area',' perimeter', 	 'circularity', 	' mu_NRL']
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'.format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'.format(lda.score(X_test, y_test)))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)))


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

#%%
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches


feature_names = ['area',' perimeter', 	 'circularity', 	' mu_NRL']
X = mass[feature_names]
y = mass['classe']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def plot_fruit_knn(X, y, n_neighbors, weights):
    X_mat = X[['height', 'width']].as_matrix()
    y_mat = y.as_matrix()
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])
    clf=LogisticRegression()
    #clf=LinearDiscriminantAnalysis()
    #clf=GaussianNB()
    #clf=SVC()
    #clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)
    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),np.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #    Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    patch0 = mpatches.Patch(color='#FF0000', label='apple')
    patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
    patch2 = mpatches.Patch(color='#0000FF', label='orange')
    patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
    plt.legend(handles=[patch0, patch1, patch2, patch3])
    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')
    plt.title("4-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
    plt.show()

plot_fruit_knn(X_train, y_train, 5, 'uniform')


