import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from activations import HermiteLayer
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Parameters
exp = '4'
dataset = 'mnist'
input_size = 28 * 28#32 * 32 #28 * 28
arch = [1000, 500, 250, 30]
epochs = 5000
activation = 'sigmoid'
batch_size = 128
lr = 1e-3
epsilon = 0.001
num_pol = 4

filename = dataset + '_' + activation + '_' + 'adam' +  str(epsilon) + '_' + 'EXP' + str(exp)


def compute_activation(activation, x):
    if activation == 'hermite':
        print('ACTIVATION: ', activation)
        x = HermiteLayer(num_pol = num_pol)(x)
        return x
    else:
        print('ACTIVATION: ', activation)
        x = Activation(activation)(x)
        return x
        
# Deep Autoencoder
input_img = Input(shape=(input_size, ))
# "encoded" is the encoded representation of the inputs
for i, n in enumerate(arch):
    if i == 0:
        encoded = Dense(n, activation=None)(input_img)
        encoded = compute_activation(activation, encoded)
    elif i == len(arch) - 1:
        encoded = Dense(n, activation='linear')(encoded)
    else:
        encoded = Dense(n, activation=None)(encoded)
        encoded = compute_activation(activation, encoded)

for i, n in enumerate(arch[::-1]):
    if i == 0:
        continue
    elif i == 1:
        decoded = Dense(n, activation=None)(encoded)
        decoded = compute_activation(activation, decoded)
    else:
        decoded = Dense(n, activation=None)(decoded)
        decoded = compute_activation(activation, decoded)

decoded = Dense(input_size, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

def L2_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

# Train to reconstruct MNIST digits

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
opt = Adam(lr = lr,
           epsilon = epsilon)
autoencoder.compile(optimizer=opt,
                    loss=L2_loss)
reduce_lr = ReduceLROnPlateau(factor=0.5,
                              patience=20)
autoencoder.summary()

# prepare input data
if dataset == 'mnist':
    (x_train, _), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
elif dataset == 'cifar10':
    (x_train, _), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train[:, :, :, 0]
    x_test = x_test[:, :, :, 0]
    
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


if activation == 'hermite':
    filename += 'num_pol' + str(num_pol)


checkpoint = ModelCheckpoint('checkpoints/' + filename + '.hdf5',
                             save_best_only=True,
                             period=500)


csv_logger = CSVLogger('results/' + filename + '.csv')

callbacks = [reduce_lr, checkpoint, csv_logger] 
autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=1,
                callbacks=callbacks)

