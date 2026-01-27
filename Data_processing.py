import scipy.io as io
import numpy as np
import pickle
import os
import pandas as pd
import hdf5storage
from matplotlib import pyplot as plt
from matplotlib import rcParams

import os

config = {
    "font.family": 'serif',  # 衬线字体
    "font.size": 14,  # 相当于小四大小
    "font.serif": ['SimSun'],  # 宋体
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)


def save_pickle_v2(path, name, x):
    with open(path + name, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
    print('save to path:', path)
    print('Save successfully!')


def load_txt_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


########################################################################################################################
Train_sample1 = []
Train_sample2 = []

data = hdf5storage.loadmat("./Raw data/DataForClassification_TimeDomain.mat")
data_gear = data["AccTimeDomain"]



X1 = data_gear[:,832:936]
Train_sample1.append(X1)
b1 = np.array(Train_sample1)
a = b1.reshape(374400)
# a1 = a[187201:374400]
xx = len(a)  # 数据长度


length = 1024
sample_n = 50  # 样本数


for j in range(sample_n):

    random_start = np.random.randint(low=0, high=(xx - 2 * length))
    X1 = a[random_start: random_start + length]
    Train_sample2.append(X1)

data_0 = np.array(Train_sample2)

path_out = './Datasets/'
os.makedirs(path_out, exist_ok=True)  # 如果没有该文件夹，则创建此文件夹
save_pickle_v2(path_out, name='C8_test.pkl', x=data_0)
