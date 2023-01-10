#########################
#鉴于WeiYi的框架比较复杂，令我欲仙欲死两周时间，我决定重新搭一个框架
#这个框架基于先进的tensorflow2.0 舍弃了很多原来有的语法
#e.g. placeholder 等
#本文描述的是多层网络结合
import tensorflow as tf
import math
import os
from Channel_generator import Channel_generator
import tensorflow.keras as keras
from tensorflow.keras import Model
from keras.layers import Input
import matplotlib.pyplot as plt
from tensorflow.linalg import matmul as matmulshit
from tensorflow.linalg import matvec as matvecshit
from keras.layers import Multiply as mulshit
from keras.callbacks import TensorBoard
from tensorflow import math

import sys
import numpy as np
class FirstNlayer(keras.layers.Layer):#第一层的单层结构
    def __init__(self, Nu,Nt, SNR,name='First_Layer'):  # block_list表示每个block有几个卷积层
        super().__init__()
        self.Nu = Nu
        self.Nt = Nt
        self.noise_var = self.Nt / self.Nu * tf.pow(10., -SNR / 10.)
    def call(self,y,H):
        A = matmulshit(H, H,transpose_a = True) + noise_var * tf.eye(2 * self.Nt)
        b = matvecshit(H, y,transpose_a = True)
        d_ = b
        r_ = d_
        x0 = tf.zeros(shape= tf.shape(b), name="x0")
        return r_,x0,d_,A
    def get_config(self):
        config = super().get_config()
        config.update({
            "Nu": self.Nu,
            "Nt": self.Nt,
            "noise_var": self.noise_var,
        })
        return config
class Nlayer(keras.layers.Layer):#单层结构
    def __init__(self, Nu,Nt, ifhyperParameter=False,name='Other_Layer'):  # block_list表示每个block有几个卷积层
        super().__init__()
        self.Nu = Nu
        self.Nt = Nt
        self.ifhyperParameter= ifhyperParameter
    def build(self,inputshape):
        if self.ifhyperParameter:#如果是超参数 则是每层的权重对应一个发射天线
            self.alpha_ = self.add_weight(shape=(2*self.Nt, 1),initializer=keras.initializers.Constant(0),trainable=True,name="alpha")
            self.beta_ = self.add_weight(shape=(2*self.Nt, 1), initializer=keras.initializers.Constant(0),trainable=True,name="beta")
        else:#如果是正常参数 则每层权重对应一个 标量
            self.alpha_ = self.add_weight(shape=(1, ), initializer=keras.initializers.Constant(0), trainable=True,name="alpha")
            self.beta_ = self.add_weight(shape=(1, ),initializer=keras.initializers.Constant(0), trainable=True,name="beta")
    def call(self, r_, xhat_,d_,A):
        if self.ifhyperParameter:#如果是超参数 则是每层的权重对应一个发射天线
            print(A.shape)
            print(d_.shape)
            print(matvecshit(A, d_).shape)
            r_ = r_ - self.alpha_ * matvecshit(A, d_)
            d_ = r_ + mulshit()([self.beta_ , d_])
            xhat_ = xhat_ + mulshit()([self.alpha_ , d_])
        else:#如果是正常参数 则每层权重对应一个 标量
            r_ = r_ - self.alpha_ * matvecshit(A, d_)
            d_ = r_ + self.beta_ * d_
            xhat_ = xhat_ + self.alpha_ * d_
        return r_, xhat_, d_, A

    def get_config(self):
        config = super().get_config()
        config.update({
            "Nu": self.Nu,
            "Nt": self.Nt,
            "ifhypyer": self.ifhyperParameter,
        })
        return config

class NNet(Model):
    #def NNet(Nu,Nt, SNR,layersNum=20,ifhyperparameter = False):#多层结构
    def __init__(self,Nu,Nt, SNR,layersNum=20,ifhyperparameter = False):#多层结构
        super().__init__(name='NNet')
        self.FirstNlayer = FirstNlayer(Nu=Nu, Nt=Nt, SNR=SNR)
        self.Nlayer = Nlayer(Nu=Nu, Nt=Nt, ifhyperParameter=ifhyperparameter)
        self.layersNum = layersNum
    def call(self, inputs):
        y = inputs[0]
        H = inputs[1]
        r_, xhat_, d_, A = self.FirstNlayer(y,H)
        for layer_id in range(1, self.layersNum):  # 第几个卷积层
            r_, xhat_, d_, A = self.Nlayer(r_, xhat_, d_, A)
        return xhat_
def loss_norm_nmse(y_true,y_pred):
    loss_ = tf.nn.l2_loss(y_pred - y_true)
    nmse_denom_ = tf.nn.l2_loss(y_true)
    return loss_ / nmse_denom_
def detect_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true-tf.sign(y_pred))/2)

#def quantizition(type = "hard"):
    #if type =="hard":

# 如下是使用说明
if __name__ == '__main__':
    ########################PART.0 参数配置############################
    #发端天线数量
    Nt = 4
    #收端天线数量
    Nu = 8
    #多径数
    L_mu = 8
    #白噪声方差
    SNR = 200
    noise_var = Nt/Nu * tf.pow(10., -SNR / 10.)

    #训练集大小
    train_size = 10000
    #测试集大小
    test_size = 5000
    #网络层数
    layersNum = 16
    MAX_EPOCHS = 30000
    BATCH_SIZE = 5  # mini-batch set size
    N_TRAIN = 10 ** 5  # training set size
    N_ITER = N_TRAIN // BATCH_SIZE  # number of iterations at each epoch
    TOL = 10 ** -4  # reference value used for convergence
    N_ITER_NO_CHANGE = 10  # reference value used for aborting training
    INIT_ETA = 5e-5  # initial learning rate

    ##########################PART.1 生成测试数据#######################
    #生成信道模型
    channel = Channel_generator(Nu = Nu,Nt = Nt,L_mu = L_mu,noise_var = noise_var)
    #初始化信道矩阵
    #生成训练集
    XCube, HCube, YCube= channel.multipleoutput(setnum=train_size,ifreal=True,ifchangeChannel=False)
    ##########################PART.2 生成网络########################
    #生成网络

    model = NNet(Nu = Nu,Nt = Nt,layersNum = layersNum,SNR=SNR,ifhyperparameter=True)
    y = Input(shape=(2*Nu), name='y')
    H = Input(shape=(2*Nu,2*Nt), name='H')
    model((y,H))

    optimizer_adm = keras.optimizers.Adam(learning_rate=INIT_ETA)  # instantiate the solver
    #编译网络，也是载入优化器和损失函数的地方
    model.compile(optimizer=optimizer_adm,
                  loss=loss_norm_nmse,
                  metrics = [detect_metric]
                  )
    #断点存储位置
    checkpoint_save_path = "./checkpoint/cep8.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=r"./mytensotboard")
    model.summary()
    for layer in model.layers:
        print(layer.name, ' is trainable? ', layer.trainable)
        #layer.trainable = False
    ##########################PART.3 训练网络########################

    history = model.fit(  # 使用model.fit()方法来执行训练过程，
        x= [YCube,HCube], y = XCube,  # 告知训练集的输入以及标签，
        batch_size=BATCH_SIZE,  # 每一批batch的大小为32，
        epochs=MAX_EPOCHS,
        validation_split=0.2,  # 从测试集中划分80%给训练集
        validation_freq=20,  # 测试的间隔次数为20
        callbacks=[cp_callback,tensorboard_callback]
    )

    # print(model.trainable_variables)
    file = open('./weights.txt', 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()
    #######################PART.4 展示各种曲线   ######################
    # 显示训练集和验证集的acc和loss曲线
    acc = history.history['detect_metric']
    val_acc = history.history['detect_metric']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()