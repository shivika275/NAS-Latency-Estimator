import pandas as pd 
import pdb
import random 

df_lat_1 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/OnePlus7T_TFlite/cifar10_inf.csv')
df = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/output_csv/NVIDIA A40/cifar10.csv')
df_perf_1 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/OnePlus7T_TFlite/cifar10_perf.csv')
df_perf_1 = df_perf_1.rename(columns={'modelname': 'Model Name'})

df_lat_2 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/Pixel6A_Tflite/latency_model_Pixel6A_cifar10.csv')
df_perf_2 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/Pixel6A_Tflite/perf_outpt_Pixel6A_cifar10.csv')
df_perf_2 = df_perf_2.rename(columns={'modelname': 'Model Name'})

df_test = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/SamsungGalaxyF62_TFlite/cifar10_test.csv')

df_all_1 = pd.merge(df_lat_1, df_perf_1, on='Model Name', how='outer')
df_all_2 = pd.merge(df_lat_2, df_perf_2, on='Model Name', how='outer')

all_idx =  df_all_1['Model Name'].tolist()
test_idx = df_test['Model Name'].tolist()
index = []
for i in all_idx:
    if not(i in test_idx):
        index.append(i)  
all_idx_2 =  df_all_2['Model Name'].tolist()
index_2=[]
for i in all_idx_2:
    if not(i in test_idx):
        index_2.append(i) 
pdb.set_trace()
random.shuffle(index)
random.shuffle(index_2)
train_model = index[:int(0.8*len(index))]
val_model =  index[int(0.8*len(index)):]  
train_model_2 = index_2[:int(0.8*len(index_2))]
val_model_2 =  index_2[int(0.8*len(index_2)):]  
# pdb.set_trace()
train_df_1 = df_all_1[df_all_1['Model Name'].isin(train_model)]
val_df_1 = df_all_1[df_all_1['Model Name'].isin(val_model)]

train_df_2 = df_all_2[df_all_2['Model Name'].isin(train_model_2)]
val_df_2 = df_all_2[df_all_2['Model Name'].isin(val_model_2)]
# index = df_all['Model Name']
name = df['arch_str']


train_model_name = []
val_model_name = []

train_model_name_2 = []
val_model_name_2 = []
for i in train_model:
    idx = name[int(i.split('-')[-1])]
    train_model_name.append(idx)
for i in val_model:
    idx = name[int(i.split('-')[-1])]
    val_model_name.append(idx)
for i in train_model_2:
    idx = name[int(i.split('-')[-1])]
    train_model_name_2.append(idx)
for i in val_model_2:
    idx = name[int(i.split('-')[-1])]
    val_model_name_2.append(idx)

# pdb.set_trace()
train_df_1['arch_str'] =train_model_name
train_df_2['arch_str'] =train_model_name_2

val_df_1['arch_str'] =val_model_name
val_df_2['arch_str'] =val_model_name_2

train_df_1.to_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/OnePlus7T_TFlite/cifar10_train.csv')
val_df_1.to_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/OnePlus7T_TFlite/cifar10_val.csv')

train_df_2.to_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/Pixel6A_Tflite/cifar10_train.csv')
val_df_2.to_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/Pixel6A_Tflite/cifar10_val.csv')