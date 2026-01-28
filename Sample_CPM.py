# pip install tensorflow_addons
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import numpy as np
import math
import os
import random
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
# from Utils.data_utils import *
# from Utils.Unet_utils import *

def LoadData_pickle(path,name,type='rb'):
  with open(path+name+'.pkl', type) as f:
          data=pickle.load(f)
  return data


# Global Settings
batch_size = 2
num_epochs = 99
learning_rate = 1e-5
img_size = 1024
img_channels = 1

faults=['C0','C1','C2','C3','C4','C5','C6', 'C7','C8']
root = './Results_AE/'     #output data
for fault in faults:
    # Build graph
    # ops.reset_default_graph()
    # Build encoder
    inputs_=layers.Input(shape=(img_size, img_channels), name="image_input")
    # 2，神经网络
    layers = tf.keras.layers
    # ### Encoder
    conv1 = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(inputs_)
    maxpool1 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv1)
    conv2 = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool1)
    maxpool2 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv2)
    conv3 = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool2)
    maxpool3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv3)
    conv4 = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool3)
    maxpool4 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv4)
    conv5 = layers.Conv1D(filters=2, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool4)
    maxpool5 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv5)
    re = tf.reshape(maxpool5, [-1, 64])
    # -----------#
    latent = layers.Dense(units=128, activation=tf.nn.relu)(re)
    # -----------#
    # ---Decoder---#
    x = layers.Dense(units=64, activation=tf.nn.relu)(re)
    x = tf.reshape(x, [-1, 32, 2])
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    rx = layers.Conv1D(filters=1, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    # print(rx.shape, inputs_.shape)
    # print('Built Encoder../')

    # print(image_input.shape, enout.shape, x_out.shape)
    # #Build model
    dcae=keras.Model(inputs_, rx)
    #
    # # Opimizer and loss function
    opt = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)
    print('Network Summary-->')
    # dcae.summary()


    dir = './Results_AE/Z/model/model_last_' + str(99) + '.ckpt'
    print('Load weights from ', dir)
    dcae.load_weights(dir)
    new_enout=tf.keras.models.Model(inputs=inputs_,outputs=latent)

    x_test = LoadData_pickle(path='./Results/', name='encoded_' + fault + '_test', type='rb')   #input
    datax_test = tf.reshape(x_test, shape=[-1, 1024, 1])
    # x_hat1=new_enout.predict(datax_test)
    # x_hat2 = dcae.call(datax_test).numpy()
    extracted_featuresx_test = new_enout.predict(datax_test)
    print(extracted_featuresx_test.shape)
    with open('./Results_AE/encoded_'+fault+'_test.pkl', 'wb') as f:
        pickle.dump(extracted_featuresx_test, f, pickle.HIGHEST_PROTOCOL)


    x_T = LoadData_pickle(path='./Results/', name='encoded_' + fault + '_train', type='rb')    #input
    data_T = tf.reshape(x_T, shape=[-1, 1024, 1])
    extracted_features_T = new_enout.predict(data_T)
    print(extracted_features_T.shape)
    with open('./Results_AE/encoded_' + fault + '_train.pkl', 'wb') as f:
        pickle.dump(extracted_features_T, f, pickle.HIGHEST_PROTOCOL)
    #




