#################################################   Import  ##################################################################
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
import os

import tensorflow as tf
from tensorflow.keras import datasets, layers,optimizers

from keras import models , layers
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from Utils import whitening
from scipy import linalg
from keras.layers import  BatchNormalization, Reshape, ReLU , Input   # Dense
from keras.layers import   Input   # Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def LoadData_pickle(path,name,type='rb'):
  with open(path+name+'.pkl', type) as f:
          data=pickle.load(f)
  return data


def save_pickle_v1(path,name,x):#
    with open(path+name, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)  #pickle.dump () 封装是一个将Python数据对象转化为字节流的过程
        print('save to path:', path)
        print('Save successfully!')


#################################################   FLOW-模型构建   ######################################################
# batch_size = 32
latent_dim = 1024
epochs_flow = 100

_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)   #alpha：负数部分的斜率。默认为 0。


x_in = Input(shape=(latent_dim,))
x = x_in


opt = tf.keras.optimizers.Adam(lr=0.0002)  #默认值 (lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)

encoder_flow = Model(x_in, x)


dir = './Results/model/model_last_' + str(epochs_flow) + '.ckpt'
print('Load weights from ', dir)
encoder_flow.load_weights(dir)
#################################################   FLOW-encoder   #######################################################

faults=['C0','C1','C2','C3','C4','C5','C6', 'C7','C8']

for fault in faults:
    path_out = './Results/'  # output data

    train_encoded = []
    test_encoded = []

    x_train  = LoadData_pickle(path='./Datasets/', name=fault + '_T', type='rb')[:,2048:3072]  # input data
    x_test = LoadData_pickle(path='./Datasets/', name=fault+ '_test', type='rb')[:,2048:3072]  # input data


    train_nice_encoded = encoder_flow.predict(x_train)  #重构数据
    train_encoded.extend(train_nice_encoded)
    train_encoded = np.array(train_encoded)


    test_nice_encoded = encoder_flow.predict(x_test)  # 重构数据
    test_encoded.extend(test_nice_encoded)
    test_encoded = np.array(test_encoded)

    save_pickle_v1(path_out,name= 'encoded_'+fault+'_train.pkl',x= train_encoded)  #output data
    save_pickle_v1(path_out, name= 'encoded_'+fault+'_test.pkl', x=test_encoded)  #output data




#####################################################################################之前#################################

