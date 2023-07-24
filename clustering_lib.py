from sklearn.cluster import KMeans 
from yellowbrick.cluster import KElbowVisualizer

## clustering functions
def clustering_kelbow_score(data, min_clusters=2, max_clusters=10, fields=[]):
    from sklearn.cluster import KMeans 
    from yellowbrick.cluster import KElbowVisualizer

    min_clusters = 2
    max_clusters = 10

    model = KMeans()

    vizualizer = KElbowVisualizer(model, k=(min_clusters,max_clusters)).fit(data[fields])
    
    vizualizer.show()
    return vizualizer.elbow_value_

def clustering_kmeans(data, clusters_num=2, fields=[]):
    cnum = clusters_num    
    k_means = KMeans(init = "k-means++", n_clusters = cnum, n_init = 12)
    xx = data.loc[:,fields]
    k_means.fit(xx)
    return k_means.labels_

def clustering_dbscan(data, min_samples=2, eps=5):
    import numpy as np
    from sklearn.cluster import DBSCAN


    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)

    return dbscan
