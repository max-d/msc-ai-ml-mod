import random 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
#import geopandas
import seaborn as sns
sns.set()
import scipy.stats as st
from scipy.stats import norm
from scipy.stats import iqr 
#from plotly import graph_objects as go
#from plotly import express as px
#%matplotlib inline


from sklearn.preprocessing import LabelEncoder, StandardScaler

# report warnings
import warnings
warnings.filterwarnings('ignore')

# import custom data preprocessing 
import preprocessing as prep

class Settings:
    """Settings is the class for EDA. 
    This class has 2 attributes:

    - distplot_attributes

    - boxplot_attributes
    """

    def distplot_attributes(self, data):
        """Plot multiple attributes using distplot"""
        
        cols = []
        
        # Iterate over columns in the data
        for i in df.columns:
            # Check if the column data type is float or int
            if df[i].dtypes == "float64" or df[i].dtypes == 'int64':
                cols.append(i)
        
        # Create a figure for the subplots
        gp = plt.figure(figsize=(15,10))
        gp.subplots_adjust(wspace=0.2, hspace=0.4)
        
        # Iterate over the selected columns
        for i in range(1, len(cols)+1):
            # Add a subplot to the figure
            ax = gp.add_subplot(3,4,i)
            
            # Plot the distribution using distplot
            sns.distplot(data[cols[i-1]], fit=norm, kde=False)
            
            # Set the title of the subplot
            ax.set_title('{}'.format(cols[i-1]))

            
    def boxplot_attributes(self, data):
        """Plot multiple attributes using boxplot"""
    
        cols = []
        
        # Iterate over columns in the data
        for i in df.columns:
            # Check if the column data type is float or int
            if df[i].dtypes == "float64" or df[i].dtypes == 'int64':
                cols.append(i)
    
        # Create a figure for the subplots
        gp = plt.figure(figsize=(20,15))
        gp.subplots_adjust(wspace=0.2, hspace=0.4)
        
        # Iterate over the selected columns
        for i in range(1, len(cols)+1):
            # Add a subplot to the figure
            ax = gp.add_subplot(3,4,i)
            
            # Plot the boxplot using boxplot
            sns.boxplot(x=cols[i-1], data=df)
            
            # Set the title of the subplot
            ax.set_title('{}'.format(cols[i-1]))

# read data from 'AB_NYC_2019.csv' file
df = pd.read_csv("AB_NYC_2019.csv")

# preprocess data
df = prep.data_preprocessing(df)

import clustering_lib as cl

# fields for clustering 
fields = ['price','number_of_reviews']

df_pn = df[((df['minimum_nights']==1) & ((df['room_type']==0) | (df['room_type']==1)) & (df['price'] < 1000))]

df_pn = df_pn[fields]

# Extract cluster labels and core samples
clusters = cl.clustering_dbscan(df_pn[fields])

labels = clusters.labels_
df_pn['clusters_pn'] = labels

core_samples = np.zeros_like(labels, dtype=bool)
core_samples[clusters.core_sample_indices_] = True

# Number of clusters in labels, ignoring noise if present
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of clusters:", n_clusters)

plt.figure(figsize=(14,10))
ax = plt.gca()
df_pn.plot(kind='scatter', x='number_of_reviews', y='price', c='clusters_pn', ax=ax, cmap=plt.get_cmap('PuBu'), colorbar=True, alpha=0.7);
plt.show()
