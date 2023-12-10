import joblib
import matplotlib.pyplot as plt
import numpy as np 
import umap
import pandas as pd
import pdb
def plot_clusters(embeddings, clusters, n_neighbors=15, min_dist=0.1):
    """
    Reduce dimensionality of best clusters and plot in 2D

    Arguments:
        embeddings: embeddings to use
        clusteres: HDBSCAN object of clusters
        n_neighbors: float, UMAP hyperparameter n_neighbors
        min_dist: float, UMAP hyperparameter min_dist for effective
                  minimum distance between embedded points

    """
    umap_data = umap.UMAP(n_neighbors=n_neighbors, 
                          n_components=2, 
                          min_dist = min_dist,  
                          #metric='cosine',
                          random_state=42).fit_transform(embeddings)

    point_size = 100.0 / np.sqrt(embeddings.shape[0])
    
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    
    result['labels'] = clusters.labels_
    result.to_csv('cluster_2.csv')

    fig, ax = plt.subplots(figsize=(14, 8))
    outliers = result[result.labels == -1]
    clustered = result[result.labels != -1]
    plt.scatter(outliers.x, outliers.y, color = 'lightgrey', s=point_size)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=point_size, cmap='jet')
    plt.colorbar()
    fig.savefig('mobile_train_2.png')
    plt.close()
    
# Load the saved HDBSCAN instance from file

# pdb.set_trace()
filename = 'hdbscan_model_train_mobile_2.sav'
loaded_hdbscan = joblib.load(filename)
loaded_embeddings = np.load('train_mobile.npy')

plot_clusters(loaded_embeddings,loaded_hdbscan)