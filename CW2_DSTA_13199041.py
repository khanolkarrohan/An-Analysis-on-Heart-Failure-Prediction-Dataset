from pandas import read_csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import scale
from pandas.plotting import scatter_matrix
import pandas as pd
from pandas import plotting
import seaborn as sns


import matplotlib.pyplot as plt; plt.rcdefaults()
%matplotlib inline

# To load the dataset and check the shape of the dataset
filename = 'heart_failure_clinical_records_dataset.csv'
data = read_csv(filename)
print(data.shape)

# Splitting the selected dimension (in X) and the target (in Y).
X = data[['ejection_fraction','serum_creatinine','serum_sodium']] # Choosen dimensions
Y = data['DEATH_EVENT'] # This is the 'target' variable which is binary
X.describe() # To learn about the dimensions

# Plotting a scatter plot matrix for the three choosen dimensions
pd.plotting.scatter_matrix(X, c = Y, cmap = plt.cm.bwr, figsize = [10,5],s=30, marker = '0')
plt.show()

# Plotting 3D plot by Varoquaux, G. for the three chosen dimensions
plot = Axes3D(plt.figure(1),elev=-150, azim=100) # Initializing the frame of the plot.
plot.scatter(X['serum_sodium'], X['serum_creatinine'], X['ejection_fraction'],c = Y, cmap = plt.cm.bwr) # Choosen dimension
plot.set_title("3D Plot of Dimensions of Interest (Target shown by colour)")  # Header of the plot
plot.set_xlabel("serum_sodium") # Lable of X axis
plot.set_ylabel("serum_creatinine")  # Lable of Y axis
plot.set_zlabel("ejection_fraction") # Lable of Z axis
plt.show()

# Using Principle component analysis
pca = PCA(n_components=3)  # specifing 3 inputs
X_PC = pca.fit_transform(X)  # performing Fit & Transform on X


# Plotting 3D plot by Varoquaux, G. for the three eigen vectors
plot = Axes3D(plt.figure(2),elev=-150, azim=100) # Initializing the frame of the plot.
plot.scatter(X_PC[:,0],X_PC[:,1],X_PC[:,2], c = Y, cmap = plt.cm.bwr) # PCA output dimension and choosing color
plot.set_title("PCA Breakdown for Three Dimensions Studied") # Header of the plot
plot.set_xlabel("First eigenvector")
plot.w_xaxis.set_ticklabels([])
plot.set_ylabel("Second eigenvector")
plot.w_yaxis.set_ticklabels([])
plot.set_zlabel("Third eigenvector")
plot.w_zaxis.set_ticklabels([])
plt.show()

# Using Variance Explaind on PCA compnents and plotting Variance explained graph
var_explained = pca.explained_variance_ratio_
plt.plot(var_explained.cumsum())
plt.xlabel('Pricipal Component') # Using lables for X axis
plt.xticks(np.arange(0,3,step=1),['PC 1','PC 2', 'PC 3'])
plt.ylabel('Cumulative Explained Variance') # Using lables for y axis
plt.title("Cumulative Variance Explained") # Header of the plot
plt.show()


# Plotting Scree graph  using variance explained
plt.plot(var_explained) # Plots Scree Plot
plt.xlabel('Pricipal Component') # Using lables for X axis
plt.xticks(np.arange(0,3,step=1),['PC 1','PC 2', 'PC 3'])
plt.ylabel('Explained Variance') # Using lables for y axis
plt.title("Scree Plot") # Header of the plot
plt.show()

# Printing PCA Components
print("PCA Components:")
print(pca.components_)