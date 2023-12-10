import pandas as pd
# from sentence_transformers import SentenceTransformer, util
import pdb 
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras import backend as K

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if not(epoch==0) and (epoch%500 ==0):  # or save after some epoch, each k-th epoch etc.
            self.model.save("model_{}.hd5".format(epoch))


def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

# pdb.set_trace()

train = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/output_csv/NVIDIA TITAN Xp/cifar10_train.csv')
val = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/output_csv/NVIDIA TITAN Xp/cifar10_val.csv')
test = pd.read_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/output_csv/NVIDIA TITAN Xp/cifar10_test.csv')


latency_train = train['latency']
latency_val = val['latency']
latency_test = test['latency']

with open('train.npy', 'rb') as f:
    embeddings_train = np.load(f)
with open('val.npy', 'rb') as f:
    embeddings_val = np.load(f)
with open('test.npy', 'rb') as f:
    embeddings_test = np.load(f)

model = Sequential()
model.add(Dense(192, input_dim = 384, activation="relu"))
model.add(Dense(96, activation="relu"))
model.add(Dense(48, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(6, activation="relu"))
model.add(Dense(3, activation="relu"))
model.add(Dense(1))

# y = Dense(64, activation="relu")(inputB)
# y = Dense(32, activation="relu")(y)
# y = Dense(4, activation="relu")(y)
# y = Model(inputs=inputB, outputs=y)

# combined = concatenate([x.output, y.output])

# model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
opt = Adam(lr=1e-4, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt,metrics=[soft_acc])
print("[INFO] training model...")
saver = CustomSaver()
history = model.fit(
	x=embeddings_train, y=latency_train,
	validation_data=(embeddings_val, latency_val),
	epochs=50000, batch_size=512,callbacks=[saver])
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
for i in range(len(embeddings_test)):
    emb = embeddings_test[i].reshape(1,embeddings_test[i].shape[0])
    preds.append(model.predict(emb)[0][0])

test['latency_prediction'] = preds
test.to_csv('/srv/hoffman-lab/flash9/apal72/NAS-Latency-Estimator/NATS-Bench/output_csv/NVIDIA TITAN Xp/output_cifar10.csv')

