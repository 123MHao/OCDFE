#################################################   Import  ##################################################################
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
import os

import tensorflow as tf
from tensorflow.keras import datasets, layers,optimizers

# from keras import models , layers
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from Utils import whitening
from scipy import linalg
from keras.layers import  BatchNormalization, Reshape, ReLU , Input   # Dense
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


#################################################   Input   ##################################################################
batch_size = 32
latent_dim = 1024
epochs_flow = 100


# Loading data
x_train  = LoadData_pickle(path='./Datasets/', name='C0_T', type='rb')
x_test = LoadData_pickle(path='./Datasets/', name='C0_test', type='rb')


#归一化处理
zca=whitening.ZCA(x = x_train)
x_train_zca=zca.apply(x_train)
x_test_zca=zca.apply(x_test)

#################################################   FLOW-模型构建   ##################################################################
_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)   #alpha：负数部分的斜率。默认为 0。

# 封装moael  定义      前向传播

class Shuffle(Layer):
#"""打乱层，提供两种方式打乱输入维度
#一种是直接反转，一种是随机打乱，默认是直接反转维度
##通过随机的方式将向量打乱，可以使信息混合得更加充分，最终的loss可以更低,
## 在NICE中，作者通过交错的方式来混合信息流（这也理论等价于直接反转原来的向量）
#"""
    def __init__(self, idxs=None, mode='reverse', **kwargs):
        super(Shuffle, self).__init__(**kwargs)
        self.idxs = idxs
        self.mode = mode
    def call(self, inputs):

        v_dim = K.int_shape(inputs)[-1]
        if self.idxs == None:
            self.idxs = list(range(v_dim))
            if self.mode == 'reverse':
                self.idxs = self.idxs[::-1]
            elif self.mode == 'random':
                np.random.shuffle(self.idxs)
        inputs = K.transpose(inputs)
        outputs = K.gather(inputs, self.idxs)
        outputs = K.transpose(outputs)
        return outputs
    def inverse(self):
        v_dim = len(self.idxs)
        _ = sorted(zip(range(v_dim), self.idxs), key=lambda s: s[1])
        reverse_idxs = [i[0] for i in _]
        return Shuffle(reverse_idxs)

class SplitVector(Layer):
    """将输入分区为两部分，交错分区
    ##就是指将每一步flow输出的两个向量h1,h2拼接成一个向量h，然后将这个向量重新随机排序。
    """
    def __init__(self, **kwargs):
        super(SplitVector, self).__init__(**kwargs)
    def call(self, inputs):
        v_dim = K.int_shape(inputs)[-1]
        inputs = K.reshape(inputs, (-1, v_dim//2, 2))
        return [inputs[:,:,0], inputs[:,:,1]]
    def compute_output_shape(self, input_shape):
        v_dim = input_shape[-1]
        return [(None, v_dim//2), (None, v_dim//2)]
    def inverse(self):
        layer = ConcatVector()
        return layer

class ConcatVector(Layer):
    """将分区的两部分重新合并
    """
    def __init__(self, **kwargs):
        super(ConcatVector, self).__init__(**kwargs)
    def call(self, inputs):
        inputs = [K.expand_dims(i, 2) for i in inputs]
        inputs = K.concatenate(inputs, 2)
        return K.reshape(inputs, (-1, np.prod(K.int_shape(inputs)[1:])))
    def compute_output_shape(self, input_shape):
        return (None, sum([i[-1] for i in input_shape]))
    def inverse(self):
        layer = SplitVector()
        return layer

class AddCouple(Layer):
    """加性耦合层
    """
    def __init__(self, isinverse=False, **kwargs):
        self.isinverse = isinverse
        super(AddCouple, self).__init__(**kwargs)
    def call(self, inputs):
        part1, part2, mpart1 = inputs
        if self.isinverse:
            return [part1, part2 + mpart1] # 逆为加
        else:
            return [part1, part2 - mpart1] # 正为减
    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1]]
    def inverse(self):
        layer = AddCouple(True)
        return layer

class Scale(Layer):

    """尺度变换层
    flow这种基于可逆变换的模型，天生就存在比较严重的维度浪费问题：输入数据明明都不是D维流形，
    但却要编码为一个D维流形，这可行吗？为了解决这个情况，
    NICE引入了一个尺度变换层，它对最后编码出来的每个维度的特征都做了个尺度变换
    """
    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[1]),
                                      initializer='glorot_normal',
                                      trainable=True)
    def call(self, inputs):
        self.add_loss(-K.sum(self.kernel)) # 对数行列式
        return K.exp(self.kernel) * inputs
    def inverse(self):
        scale = K.exp(-self.kernel)
        return Lambda(lambda x: scale * x)

def build_basic_model_v2(v_dim):
    """基础模型，即加性耦合层中的m
    """
    _in = Input(shape=(v_dim,))  ## _in是输入， _是输出
    _in = BatchNormalization()(_in)
    _ = _in
    _ = Dense(v_dim, activation=None)(_)
    for i in range(5):
        _ = Dense(1000, activation=None, kernel_regularizer=tf.keras.regularizers.l1(l=0.1))(_)   #加入L1正则化，防止过拟合
        _ = BatchNormalization()(_)
        _ = _leaky_relu(_)
    _ = Dense(v_dim, activation=_leaky_relu)(_)
    out = _
    return Model(_in, out)

shuffle1 = Shuffle()
shuffle2 = Shuffle()
shuffle3 = Shuffle()
shuffle4 = Shuffle()

split = SplitVector()
couple = AddCouple()
concat = ConcatVector()
scale = Scale()

basic_model_1 = build_basic_model_v2(latent_dim//2)
basic_model_2 = build_basic_model_v2(latent_dim//2)
basic_model_3 = build_basic_model_v2(latent_dim//2)
basic_model_4 = build_basic_model_v2(latent_dim//2)
#编码器的结构
x_in = Input(shape=(latent_dim,))
x = x_in

# 给输入加入负噪声  防止过拟合
x = Lambda(lambda s: K.in_train_phase(s-0.01*K.random_uniform(K.shape(s)), s))(x)

x = shuffle1(x)
x1,x2 = split(x)
mx1 = basic_model_1(x1)
x1, x2 = couple([x1, x2, mx1])
x = concat([x1, x2])

x = shuffle2(x)
x1,x2 = split(x)
mx1 = basic_model_2(x1)
x1, x2 = couple([x1, x2, mx1])
x = concat([x1, x2])

x = shuffle3(x)
x1,x2 = split(x)
mx1 = basic_model_3(x1)
x1, x2 = couple([x1, x2, mx1])
x = concat([x1, x2])

x = shuffle4(x)
x1,x2 = split(x)
mx1 = basic_model_4(x1)
x1, x2 = couple([x1, x2, mx1])
x = concat([x1, x2])

x = scale(x)

# model.compile 配置

# 编码器模型
opt = tf.keras.optimizers.Adam(lr=0.0002)  #默认值 (lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)

encoder_flow = Model(x_in, x)
encoder_flow.summary()

encoder_flow.compile(loss=lambda y_true,y_pred:0.5 * K.sum(y_pred**2, 1), #loss函数写成一个Lambda层了，即该层的输出就是模型的loss值，也就是模型的预测值 y_pred.
                optimizer=opt)

history_flow = encoder_flow.fit(x_train_zca,
                                x_train_zca,
                                batch_size = batch_size,
                                validation_data = (x_test_zca, x_test_zca),
                                epochs = epochs_flow)



save_dir='./Results/model/'
os.makedirs(save_dir, exist_ok=True)
encoder_flow.save_weights(save_dir+'model_last_'+str(epochs_flow)+'.ckpt')

# loss图可视化
plt.figure()
plt.plot(history_flow.history['loss'], linestyle='solid', marker='o', linewidth=1.5, c='b',label="Training loss")
plt.plot(history_flow.history['val_loss'], linestyle=(0, (3, 1, 1, 1)), marker='^', linewidth=2, c='r', label="Validation loss")

plt.title('Model_flow loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


