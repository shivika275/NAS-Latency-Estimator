import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np 

train = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/OnePlus7T_TFlite/cifar10_train.csv')
train_2 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/Pixel6A_Tflite/cifar10_train.csv')
val = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/OnePlus7T_TFlite/cifar10_val.csv')
val_2 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/Pixel6A_Tflite/cifar10_val.csv')
test = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/SamsungGalaxyF62_TFlite/cifar10_test.csv')

archs_train = train['arch_str'].tolist() + train_2['arch_str'].tolist()


archs_val = val['arch_str'].tolist() + val_2['arch_str'].tolist()


archs_test = test['arch_str']

pdb.set_trace()
print('MODEL')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print('EMBEDDINGS')
embeddings_train = model.encode(archs_train)
embeddings_val = model.encode(archs_val)
embeddings_test = model.encode(archs_test)

with open('train_mobile_2.npy', 'wb') as f:
    np.save(f,embeddings_train)

with open('val_mobile_2.npy', 'wb') as f:
    np.save(f,embeddings_val)

with open('test_mobile_2.npy', 'wb') as f:
    np.save(f,embeddings_test)