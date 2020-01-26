import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import Callback
from scipy.misc import factorial2
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
epochs = 100
learning_rate = 0.1
batch_size = 128

class Hermite:
    def __init__(self):
        self.h = []

        def h0(x):    return 1
        self.h.append(h0)

        def h1(x):    return x
        self.h.append(h1)
        
        def h2(x):    return (tf.pow(x, 2) - 1)/np.sqrt(np.math.factorial(2))
        self.h.append(h2)

        def h3(x):    return (tf.pow(x, 3) - 3*x)/np.sqrt(np.math.factorial(3))
        self.h.append(h3)

        def h4(x):    return (tf.pow(x, 4) - 6*tf.pow(x, 2) + 3)/np.sqrt(np.math.factorial(4))
        self.h.append(h4)

        def h5(x): return (tf.pow(x, 5) - 10*tf.pow(x, 3) + 15*x)/np.sqrt(np.math.factorial(5))
        self.h.append(h5)

        def h6(x): return (tf.pow(x, 6) - 15*tf.pow(x, 4) + 45*tf.pow(x,2) - 15)/np.sqrt(np.math.factorial(6))
        self.h.append(h6)

        def h7(x): return (tf.pow(x,7) - 21*tf.pow(x,5) + 105*tf.pow(x,3) - 105*x)/np.sqrt(np.math.factorial(7))
        self.h.append(h7)

        def h8(x): return (tf.pow(x,8) - 28*tf.pow(x,6) + 210*tf.pow(x,4) - 420*tf.pow(x,2) + 105)/np.sqrt(np.math.factorial(8))
        self.h.append(h8)

        def h9(x): return (tf.pow(x,9) - 36*tf.pow(x,7) + 378*tf.pow(x,5) - 1260*tf.pow(x,3) + 945*x)/np.sqrt(np.math.factorial(9))
        self.h.append(h9)

        def h10(x): return (tf.pow(x,10) - 45*tf.pow(x,8) + 630*tf.pow(x,6) - 3150*tf.pow(x,4) + 4725*tf.pow(x,2) - 945)/np.sqrt(np.math.factorial(10))
        self.h.append(h10)        


    def get_initializations(self, num_pol = 5, copy_fun = 'relu'):
        k = []
        if copy_fun == 'relu':
            for n in range(num_pol):
                if n == 0:
                    k.append(1.0/np.sqrt(2*np.pi))
                elif n == 1:
                    k.append(1.0/2)
                elif n == 2:
                    k.append(1.0/np.sqrt(4*np.pi))
                elif n > 2 and n % 2 == 0:
                    c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.pi*np.math.factorial(n))
                    k.append(c)
                elif n >= 2 and n % 2 != 0:
                    k.append(0.0)
        return k

    def hermite(self, x, k, num_pol = 5):
        evals = 0.0
        print('NUMPOL',  num_pol)
        for i in range(num_pol):
            evals += k[i]*self.h[i](x)
        return evals

class HermiteLayer(Layer):
    def __init__(self, num_pol= 5, **kwargs):
        self.num_pol = num_pol
        super(HermiteLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.acti = Hermite()
        k = self.acti.get_initializations(num_pol = self.num_pol)
        self.kernel = self.add_weight(name='Koeff',
                                      shape = tf.TensorShape(self.num_pol),
                                      initializer=Constant(k),
                                      trainable=True)
        
        # Be sure to call this at the end
        super(HermiteLayer, self).build(input_shape)

    def call(self, inputs):
        return self.acti.hermite(inputs, self.kernel, num_pol = self.num_pol)

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        logs['test_acc'] = acc
        logs['test_loss'] = loss
        print('Testing loss: {}, acc: {}'.format(loss, acc))
            
            
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("Shape of training data:")
print(X_train.shape)
print(y_train.shape)
print("Shape of test data:")
print(X_test.shape)
print(y_test.shape)

# Transform label indices to one-hot encoded vectors

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Transform images from (32,32,3) to 3072-dimensional vectors (32*32*3)

X_train = np.reshape(X_train,(50000,3072))
X_test = np.reshape(X_test,(10000,3072))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalization of pixel values (to [0-1] range)

X_train /= 255
X_test /= 255

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

# ----- HERMITE 3 hermites-----
modelh3 = Sequential()
modelh3.add(Dense(256, input_dim=3072, use_bias=False))
modelh3.add(BatchNormalization())
modelh3.add(HermiteLayer(num_pol = 3))
modelh3.add(Dense(256, use_bias=False))
modelh3.add(BatchNormalization())
modelh3.add(HermiteLayer(num_pol = 3))
modelh3.add(Dense(10, activation='softmax', use_bias=False))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=learning_rate, decay=0.0, momentum=0.0, nesterov=False)
modelh3.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history_hermiteh3 = modelh3.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.0, callbacks=[TestCallback((X_test, y_test))])
# -------------------

# ----- HERMITE 5 hermites-----
modelh5 = Sequential()
modelh5.add(Dense(256, input_dim=3072, use_bias=False))
modelh5.add(BatchNormalization())
modelh5.add(HermiteLayer(num_pol = 5))
modelh5.add(Dense(256, use_bias=False))
modelh5.add(BatchNormalization())
modelh5.add(HermiteLayer(num_pol = 5))
modelh5.add(Dense(10, activation='softmax', use_bias=False))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=learning_rate, decay=0.0, momentum=0.0, nesterov=False)
modelh5.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history_hermiteh5 = modelh5.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.0, callbacks=[TestCallback((X_test, y_test))])
# -------------------

# ----- HERMITE 10 hermites-----
modelh8 = Sequential()
modelh8.add(Dense(256, input_dim=3072, use_bias=False))
modelh8.add(BatchNormalization())
modelh8.add(HermiteLayer(num_pol = 8))
modelh8.add(Dense(256, use_bias=False))
modelh8.add(BatchNormalization())
modelh8.add(HermiteLayer(num_pol = 8))
modelh8.add(Dense(10, activation='softmax', use_bias=False))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=learning_rate, decay=0.0, momentum=0.0, nesterov=False)
modelh8.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history_hermiteh8 = modelh8.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.0, callbacks=[TestCallback((X_test, y_test))])
# -------------------

# ------- RELU -------
model = Sequential()
model.add(Dense(256, input_dim=3072))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=learning_rate, decay=0.0, momentum=0.0, nesterov=False)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history_relu = model.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.0, callbacks=[TestCallback((X_test, y_test))])
# ------------------

import pandas as pd
df_loss = pd.DataFrame(data={
    'h3':history_hermiteh3.history['loss'],
    'h5':history_hermiteh5.history['loss'],
    'h8':history_hermiteh8.history['loss'],
    'relu':history_relu.history['loss'],
})
df_loss.to_csv('stats_loss.csv')

df_train = pd.DataFrame(data={
    'h3':history_hermiteh3.history['acc'],
    'h5':history_hermiteh5.history['acc'],
    'h8':history_hermiteh8.history['acc'],
    'relu':history_relu.history['acc'],
})
df_train.to_csv('stats_train_acc.csv')

df_test = pd.DataFrame(data={
    'h3':history_hermiteh3.history['test_acc'],
    'h5':history_hermiteh5.history['test_acc'],
    'h8':history_hermiteh8.history['test_acc'],
    'relu':history_relu.history['test_acc'],
})
df_test.to_csv('stats_test_acc.csv')
