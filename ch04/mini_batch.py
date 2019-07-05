# coding: utf-8
import sys
import os

import numpy as np
import pickle
from dataset.mnist import load_mnist
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
