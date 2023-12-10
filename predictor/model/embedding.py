import pandas as pd 
from sentence_transformers import SentenceTransformer, util
import pdb
import random
from functools import partial
import numpy as np
# import matplotlib.pyplot as plt
import hdbscan
import umap
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm
import joblib


def score_clusters(clusters, prob_threshold = 0.05):
    """
    Returns the label count and cost of a given cluster supplied from running hdbscan
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)
    
    return label_count, cost



def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components, 
                      min_cluster_size,
                      random_state = None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """
    
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric='cosine', 
                                random_state=random_state)
                            .fit_transform(message_embeddings))
    print(umap_embeddings.shape)

    clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                               metric='euclidean', 
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters


def random_search(embeddings, space, num_evals):
    """
    Randomly search hyperparameter space and limited number of times 
    and return a summary of the results
    """
    
    results = []
    
    for i in tqdm(range(num_evals), total = num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        
        clusters = generate_clusters(embeddings, 
                                     n_neighbors = n_neighbors, 
                                     n_components = n_components, 
                                     min_cluster_size = min_cluster_size, 
                                     random_state = 42)
    
        label_count, cost = score_clusters(clusters, prob_threshold = 0.05)
                
        results.append([i, n_neighbors, n_components, min_cluster_size, 
                        label_count, cost])
    
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components', 
                                               'min_cluster_size', 'label_count', 'cost'])
    
    return result_df.sort_values(by='cost'), clusters



root_csv = '/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/output_csv/NVIDIA TITAN Xp/cifar100.csv'

df = pd.read_csv(root_csv)
archs = df['arch_str']
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# embeddings = model.encode(archs)

# embeddings = model.load
space = {
        "n_neighbors": range(5,9),
        "n_components": range(4,10),
        "min_cluster_size": range(100,200),
        "random_state": 42
    }

with open('train_mobile.npy', 'rb') as f:
    embeddings = np.load(f)

random_use, clusters = random_search(embeddings, space, 50)
# pdb.set_trace()
print('RANDOM_USE')
print(random_use)
filename = 'hdbscan_model_train_mobile_2.sav'
joblib.dump(clusters, filename)
# print(clusters)
# plot_clusters(embeddings, clusters)