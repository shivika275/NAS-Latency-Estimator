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
from keras.utils.vis_utils import plot_model
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if not(epoch==0) and (epoch%1000 ==0):  # or save after some epoch, each k-th epoch etc.
            self.model.save("model_hw{}.hd5".format(epoch))


def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))



train_1 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/OnePlus7T_TFlite/cifar10_train.csv')
val_1 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/OnePlus7T_TFlite/cifar10_val.csv')
train_2 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/Pixel6A_Tflite/cifar10_train.csv')
val_2 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/Pixel6A_Tflite/cifar10_val.csv')

test = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/SamsungGalaxyF62_TFlite/cifar10_test.csv')

idx = [i for i in range(len(train_1)+len(train_2))]
random.shuffle(idx)

latency_train_1 = train_1['Inference Average Time (us)'].tolist() + train_2['Inference Average Time (us)'].tolist()
latency_train = [latency_train_1[i] for i in idx]
latency_val = val_1['Inference Average Time (us)'].tolist() + val_2['Inference Average Time (us)'].tolist()
latency_test = test['Inference Average Time (us)'].tolist()



with open('train_mobile.npy', 'rb') as f:
    embeddings_train_1 = np.load(f)
with open('val_mobile.npy', 'rb') as f:
    embeddings_val = np.load(f)
with open('test_mobile.npy', 'rb') as f:
    embeddings_test = np.load(f)


embeddings_train = [embeddings_train_1[i] for i in idx]

columns_to_drop = ['Unnamed: 0','Model Name','Initialization Time (ms)','First Inference Time (us)','Inference Average Time (us)','Warmup Average Time (us)','arch_str']

# pdb.set_trace()
scale = StandardScaler()
train_11 = train_1.drop(columns=columns_to_drop)
train_22 = train_2.drop(columns=columns_to_drop)
common_columns = set(train_11.columns).intersection(train_22.columns)
columns_to_drop1 = [col for col in columns_to_drop if col in common_columns]
hw_train_1 =  pd.concat([train_11.drop(columns=columns_to_drop1), train_22.drop(columns=columns_to_drop1)], axis=0)
hw_train_1.reset_index(drop=True, inplace=True)
hw_train = hw_train_1.iloc[idx]
hw_train.reset_index(drop=True, inplace=True)


print(np.any(np.isnan(hw_train)))
scale.fit(hw_train)
hw_train = scale.transform(hw_train)
hw_val =  pd.concat([val_1.drop(columns=columns_to_drop), val_2.drop(columns=columns_to_drop)], axis=0)
hw_val.reset_index(drop=True, inplace=True)
hw_val = scale.transform(hw_val)
columns_to_drop_test = ['Unnamed: 0','Unnamed: 0.1','Model Name','Initialization Time (ms)','First Inference Time (us)','Inference Average Time (us)','Warmup Average Time (us)','arch_str','latency_prediction']
hw_test = test.drop(columns=columns_to_drop_test)




# model_emb = Sequential()
# model_emb.add(Dense(192, input_dim = 384, activation="relu"))
# model_emb.add(Dense(96, activation="relu"))
# model_emb.add(Dense(48, activation="relu"))
# model_emb.add(Dense(24, activation="relu"))
# model_emb.add(Dense(12, activation="relu"))
# model_emb.add(Dense(6, activation="relu"))

# model_hw = Sequential()
# model_hw.add(Dense(320, input_dim=10, activation='relu'))
# model_hw.add(Dense(384, activation='relu'))
# model_hw.add(Dense(352, activation='relu'))
# model_hw.add(Dense(448, activation='relu'))
# model_hw.add(Dense(160, activation='relu'))
# model_hw.add(Dense(160, activation='relu'))
# model_hw.add(Dense(32, activation='relu'))
# model_hw.add(Dense(16, activation='relu'))
# model_hw.add(Dense(6, activation='relu'))
# # model_hw.add(Dense(1))

# combined_input = concatenate([model_emb.output, model_hw.output])


# x = Dense(6, activation="relu")(combined_input)
# x = Dense(3, activation="relu")(combined_input)
# x = Dense(1, activation="linear")(x)
# pdb.set_trace()
# model = Model(inputs=[model_emb.output, model_hw.output], outputs=x)


input_emb = Input(shape=(384,))
input_hw = Input(shape=(10,))

# Model for model_emb
x = Dense(192, activation="relu",kernel_regularizer=regularizers.l2(0.01))(input_emb)
x = Dense(96, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)
x = Dense(48, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)
x = Dense(24, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)
x = Dense(12, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)
output_emb = Dense(6, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)

# pdb.set_trace()
# Model for model_hw
y = Dense(192, activation='relu')(input_hw)
y = Dense(96, activation='relu')(y)
y = Dense(48, activation='relu')(y)
y = Dense(24, activation='relu')(y)
y = Dense(12, activation='relu',kernel_regularizer=regularizers.l2(0.01))(y)
y = Dense(8, activation='relu',kernel_regularizer=regularizers.l2(0.01))(y)
output_hw = Dense(6, activation='relu',kernel_regularizer=regularizers.l2(0.01))(y)

# Concatenate outputs from both models
combined = concatenate([output_emb, output_hw])

# Further layers after concatenation
z = Dense(6, activation="relu",kernel_regularizer=regularizers.l2(0.01))(combined)
z = Dense(3, activation="relu",kernel_regularizer=regularizers.l2(0.01))(z)
output = Dense(1, activation="linear")(z)
# pdb.set_trace()
# Define the model with two inputs and one output
model = Model(inputs=[input_emb, input_hw], outputs=output)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# y = Dense(64, activation="relu")(inputB)
# y = Dense(32, activation="relu")(y)
# y = Dense(4, activation="relu")(y)
# y = Model(inputs=inputB, outputs=y)

# combined = concatenate([x.output, y.output])

model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
opt = Adam(lr=1e-3, decay=1e-3 / 200,clipnorm=100.0)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt,metrics=[soft_acc])
print("[INFO] training model...")
saver = CustomSaver()
history = model.fit(
	x=[embeddings_train,hw_train], y=latency_train,
	validation_data=([embeddings_val,hw_val], latency_val),
	epochs=10000, batch_size=2048,callbacks=[saver])
plt.plot(history.history['soft_acc'])
plt.plot(history.history['val_soft_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('trainval_accuracy.png')
plt.close()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('trainval_loss.png')
plt.close()
# model.save_weights("model_arch.h5")
# pdb.set_trace()
preds = []
hw_test = scale.transform(hw_test)
preds = model.predict([embeddings_test,hw_test])
# for i in range(len(embeddings_test)):
#     emb = embeddings_test[i].reshape(1,embeddings_test[i].shape[0])
#     preds.append(model.predict(emb)[0][0])
flattened_list = [num for sublist in preds for num in sublist]
test['latency_prediction'] = flattened_list
test.to_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/mobile/SamsungGalaxyF62_TFlite/cifar10_test.csv')

