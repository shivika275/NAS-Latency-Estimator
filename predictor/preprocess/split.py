import pandas as pd
root_csv = '/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/output_csv/NVIDIA TITAN Xp/cifar10.csv'
df = pd.read_csv(root_csv)
shuffled_df = df.sample(frac=1).reset_index(drop=True)
train = shuffled_df.iloc[:int(0.7*len(shuffled_df))]
val = shuffled_df.iloc[int(0.7*len(shuffled_df)):int(0.9*len(shuffled_df))]
test = shuffled_df.iloc[int(0.9*len(shuffled_df)):]

train.to_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/output_csv/NVIDIA TITAN Xp/cifar10_train.csv')
val.to_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/output_csv/NVIDIA TITAN Xp/cifar10_val.csv')
test.to_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/output_csv/NVIDIA TITAN Xp/cifar10_test.csv')