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

print("records before data preprocessing: " + str(len(df.index)))

# preprocess data
df = prep.data_preprocessing(df)

print("records after data preprocessing: " + str(len(df.index)))

### ---------------------- clustering ---------------------- 
import clustering_lib as cl_lib

print(df.head())
print(df.info())
print(df.columns)

## cluster: 
fields = ['price','number_of_reviews']

df1 = df[((df['minimum_nights']==1) & (df['room_type']==1))]

estimated_clusters_num = cl_lib.clustering_kelbow_score(df1, fields=fields)
labels = cl_lib.clustering_kmeans(df1, estimated_clusters_num, fields)
df1["clusters_6"] = labels

means_5 = df1.groupby('clusters_6')[fields].mean()

print(str(means_5))

plt.figure(figsize=(14,10))
ax = plt.gca()
df1.plot(kind='scatter', x='number_of_reviews', y='price', c='clusters_6', ax=ax, cmap=plt.get_cmap('PuBu'), colorbar=True, alpha=0.7);
plt.show()

