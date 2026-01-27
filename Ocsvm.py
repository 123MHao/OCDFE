import sklearn
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np


#################################################   Input   ############################################################

dir='./Results_AE/'
path_train_0 = dir+'encoded_C0_T.pkl'



with open(path_train_0,'rb') as f:
        X_train_00 = pickle.load(f)   #带标签数据读取 X_train_0 = pickle.load(f)[0]
        X_train_0 =   X_train_00.reshape(-1,128)

 # 标准化
# des = X_train_0.std(axis=0)
# media = X_train_0.mean(axis=0)
# X_train_0 = (X_train_0 - media) / des


#################################################   # 进行Ocsvm训练   ####################################################

clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')  #  gamma=0.00001 gamma='auto' gamma=0.00001//  nu=0.9, kernel='rbf', gamma='auto'  (nu=nu, kernel='rbf', gamma='auto')
clf.fit(X_train_0)
y_train_pred = clf.predict(X_train_0)
# 测试阶段
faults=['C0','C1','C2','C3','C4','C5','C6', 'C7','C8']
accum_percent = []
ac = []
for fault in faults:
    # # 依次送入每类故障数据
    y_pred_test = []
    n_error_test1 = []
    path_test = './Results_AE/encoded_'+fault+'_test.pkl'  #特征提取


    with open(path_test, 'rb') as f:
        X_test = pickle.load(f)

    # 标准化
    # X_test = (X_test - media) / des
    y_pred_test = clf.predict(X_test)

    # 进行比较
    if fault == 'C0':
        n_error_test = y_pred_test[y_pred_test == -1].size     #异常数据
    else:
        n_error_test = y_pred_test[y_pred_test == 1].size      #正常数据

    percent = n_error_test / len(X_test)
    print('% test errors condition', fault, ':', percent)
    accum_percent.append(1 - percent)
    n_error_test1.append(n_error_test)
    # print(accum_percent)
    accuracy = np.array(accum_percent).mean()
print('Avg. Accuracy:', accuracy)

# print(confusion_matrix(y_test_real,y_pred_test))

ac.append(accuracy)
ac = np.array(ac)





