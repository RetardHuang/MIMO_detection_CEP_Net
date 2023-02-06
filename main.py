import numpy as np
import tensorflow as tf
import os
import matplotlib
import CEPNet_New
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from Channel_generator import Channel_generator
from CEPNet_New import NNet
#设置随机种子
np.random.seed(1)
# 当数组元素比较多的时候，如果输出该数组，那么会出现省略号,这一句让输出变为全部
np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    ########################PART.0 参数配置############################
    #发端天线数量
    Nt = 8
    #收端天线数量
    Nu = 16
    #如果更改天线数量，就删除checkpoint 文件夹
    #多径数
    L_mu = 8
    #训练集大小
    train_size = 100000
    #测试集大小
    test_size = 50
    #网络层数
    layersNum = 16
    SNR = 30
    BATCH_SIZE = 1  # mini-batch set size
    model = NNet(Nu = Nu,Nt = Nt,layersNum = layersNum,SNR=SNR,ifhyperparameter=True)
    #history = model.multiple_Frozen_compile_fit(train_size=train_size,each_layer_batchsize = BATCH_SIZE,eachlayer_epochs=1, validation_split=0.01,  validation_freq=120)
    #history = model.normal_compile_fit(train_size=train_size,batchsize=BATCH_SIZE, epochs=1, validation_split=0.01,  validation_freq=120)
    #model.save_weights("my_model.h5")
    model.load_weights("my_model.h5")
    SER = model.multiple_predict(test_size = 1000,SNR =100,iflogSER=False)
    print("SER is",tf.get_static_value(SER))
    pass