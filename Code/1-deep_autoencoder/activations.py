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
                    #k.append(0.0)
                elif n == 1:
                    k.append(1.0/2)
                    #k.append(0.0)
                elif n == 2:
                    k.append(1.0/np.sqrt(4*np.pi))
                    #k.append(0.0)
                elif n > 2 and n % 2 == 0:
                    c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.pi*np.math.factorial(n))
                    k.append(c)
                    #k.append(0.0)
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
        
        # Be sure to call this at the end[<0;133;49M
        super(HermiteLayer, self).build(input_shape)

    def call(self, inputs):
        return self.acti.hermite(inputs, self.kernel, num_pol = self.num_pol)

    
