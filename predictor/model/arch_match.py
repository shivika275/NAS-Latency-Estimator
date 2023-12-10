import pandas as pd
# from sentence_transformers import SentenceTransformer, util
import pdb 
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, concatenate, Input
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras import backend as K
from sklearn.preprocessing import StandardScaler
import random 
from keras import regularizers
with open('train_mobile.npy', 'rb') as f:
    embeddings_train = np.load(f)

with open('test_mobile.npy', 'rb') as f:
    embeddings_test = np.load(f)


clusters = pd.read_csv('cluster.csv')
df = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/OnePlus7T_TFlite/cifar10_train.csv')
df_test = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/SamsungGalaxyF62_TFlite/cifar10_test.csv')
threshold = len(df)-1
estimate = []
# pdb.set_trace()
for emb in tqdm(embeddings_test,total=len(embeddings_test)):
    similarities = cosine_similarity(embeddings_train, [emb])
    # Find the index of the embedding with the highest similarity score
    idx = np.argmax(similarities)
    cluster = clusters['labels'].iloc[idx]
    indices = clusters.index[clusters['labels'] == cluster].tolist()
    filtered_indices = [x for x in indices if x <= threshold]
    values = df.loc[filtered_indices, 'Inference Average Time (us)']
    
    estimate.append((max(values) - min(values))/2)
df_test['cluster_estimate'] = estimate

df_test.to_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/SamsungGalaxyF62_TFlite/cifar10_test_cluster.csv')




    



