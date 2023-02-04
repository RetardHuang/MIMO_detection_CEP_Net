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
    train_size = 10000
    #测试集大小
    test_size = 50
    #网络层数
    layersNum = 30
    SNR = 30
    BATCH_SIZE = 1  # mini-batch set size
    model = NNet(Nu = Nu,Nt = Nt,layersNum = layersNum,SNR=SNR,ifhyperparameter=True)
    history = model.normal_compile_fit(  # 使用model.fit()方法来执行训练过程，
        train_size=train_size,
        batchsize=BATCH_SIZE,  #
        epochs=20,
        validation_split=0.01,
        validation_freq=1,  # 测试的间隔次数为20
    )
    SER = model.multiple_predict(test_size = 1000,SNR = 30,iflogSER=False)
    print("SER is",tf.get_static_value(SER))
    pass