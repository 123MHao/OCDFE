import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.manifold import TSNE


def LoadData_pickle(path,name,type='rb'):
  with open(path+name+'.pkl', type) as f:
          data=pickle.load(f)
  return data
#visiualize
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sn.color_palette("hls", 9))

    # We create a scatter plot.
    f = plt.figure(figsize=(10, 10) )
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    # plt.grid(c='r')
    ax.axis('on')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(9):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

############################################       input              ######################################################

y0=np.ones([300])*0
y1=np.ones([50])*1
y2=np.ones([50])*2
y3=np.ones([50])*3
y4=np.ones([50])*4
y5=np.ones([50])*5
y6=np.ones([50])*6
y7=np.ones([50])*7
y8=np.ones([50])*8


y_test = np.hstack([y0, y1, y2, y3,y4 ,y5, y6, y7, y8]).astype(np.int64)

dir = './Results/'


x0  = LoadData_pickle(path=dir, name='encoded_C0_test', type='rb')
x1  = LoadData_pickle(path=dir, name='encoded_C1_test', type='rb')
x2  = LoadData_pickle(path=dir, name='encoded_C2_test', type='rb')
x3  = LoadData_pickle(path=dir, name='encoded_C3_test', type='rb')
x4  = LoadData_pickle(path=dir, name='encoded_C4_test', type='rb')
x5  = LoadData_pickle(path=dir, name='encoded_C5_test', type='rb')
x6  = LoadData_pickle(path=dir, name='encoded_C6_test', type='rb')
x7  = LoadData_pickle(path=dir, name='encoded_C7_test', type='rb')
x8  = LoadData_pickle(path=dir, name='encoded_C8_test', type='rb')


X_test = np.vstack([x0, x1, x2, x3,x4 ,x5, x6, x7, x8])


############################################                                ############################################

sn.set_style('whitegrid')
sn.set_palette('muted')
sn.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

digits_proj = TSNE().fit_transform(X_test)


scatter(digits_proj, y_test)
plt.savefig('./Figures/Fig.png')
plt.show()


