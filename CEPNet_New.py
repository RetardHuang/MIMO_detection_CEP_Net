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
        b = matmulshit(H, y,transpose_a = True)
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
            r_ = r_ - tf.multiply(self.alpha_ , matmulshit(A, d_))
            d_ = r_ + tf.multiply(self.beta_ , d_)
            xhat_ = xhat_ + tf.multiply(self.alpha_ , d_)
        else:#如果是正常参数 则每层权重对应一个 标量
            r_ = r_ - self.alpha_ * matmulshit(A, d_)
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

def loss_norm_nmse(y_true,y_pred):
    loss_ = tf.nn.l2_loss(y_pred - y_true)
    nmse_denom_ = tf.nn.l2_loss(y_true)
    return loss_ / nmse_denom_
def detect_metric(y_true, y_pred):
    #tf.print("y_true:", y_true, summarize=-1,output_stream=sys.stdout)
    #tf.print("y_pred:", y_pred, summarize=-1,output_stream=sys.stdout)
    return tf.reduce_mean(tf.abs(y_true-tf.sign(y_pred))/2)
#我在这里定义一个子类，这个子类的方法中间就是冻结层的训练方法
#为什么不在这个子类里面直接定义网络结构呢，请看我下一段话！
class multi_frozenlayer_model(Model):
    def multiple_Frozen_compile_fit(self,train_size,
                                    each_layer_batchsize,
                                    eachlayer_epochs,
                                    validation_split,
                                    validation_freq
                                    ):
        # 生成信道模型
        channel = Channel_generator(Nu=Nu, Nt=Nt, L_mu=L_mu, noise_var=noise_var)
        # 初始化信道矩阵
        optimizer_adm = keras.optimizers.Adam(learning_rate=INIT_ETA)  # instantiate the solver
        checkpoint_save_path = "./checkpoint/cep8.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model.load_weights(checkpoint_save_path)

        #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         #save_weights_only=True,
                                                         #save_best_only=True)
        tensorboard_callback = TensorBoard(log_dir=r"./mytensotboard")
        model.summary()
        # 全部冻结
        for layer in self.layers:
            layer.trainable = False
        # 依次开启层，然后训练
        history=[]
        for layer in self.layers :
            if len(layer.non_trainable_variables)!=0:
                #
                print('Now, training layer:',layer)
                #打开该层
                layer.trainable = True
                # 生成训练集
                XCube, HCube, YCube = channel.multipleoutput(setnum=train_size, ifreal=True, ifchangeChannel=True)
                # 编译网络，也是载入优化器和损失函数的地方
                self.compile(optimizer=optimizer_adm,
                              loss=loss_norm_nmse,
                              metrics=[detect_metric]
                              )
                # 断点存储位置
                history.append(model.fit(  # 使用model.fit()方法来执行训练过程，
                    x= [YCube,HCube], y = XCube,  # 告知训练集的输入以及标签，
                    batch_size=each_layer_batchsize,  # 每一批batch的大小为32，
                    epochs=eachlayer_epochs,
                    validation_split=validation_split,  # 从测试集中划分80%给训练集
                    validation_freq=validation_freq,  # 测试的间隔次数为20
                    callbacks=[tensorboard_callback]
                )
                )
                #该层关闭
                layer.trainable = False
            else:
                pass
        return history
#我在这里吐槽下，Model的子类构建方法弄得跟屎一样！你根本无法采用批学习
#如果你在modelclass里面定义网络结构，会出现 权重无法点乘的傻逼问题！！！
#而且，乘法也要注意，如果时矩阵乘向量，必须要采用linalg.matvec这个函数
#如果还是你不知道怎么回事的话，你就运行一下CEPNet_modelclass.py吧
#####真尼玛傻逼！！！！#####
#keras，我真尼玛后悔用，早知道用tf1的框架了！
#所以这里我们在类外部构建网络！
def NNet(Nu,Nt, SNR,layersNum=20,ifhyperparameter = False):#多层结构
    H = Input(shape=(2*Nu,2*Nt), name='H')
    y = Input(shape=(2*Nu,1), name='y')
    r_, xhat_, d_, A = FirstNlayer(Nu=Nu, Nt=Nt, SNR=SNR)(y,H)
    for layer_id in range(1, layersNum):  # 第几个卷积层
        r_, xhat_, d_, A = Nlayer(Nu=Nu, Nt=Nt, ifhyperParameter=ifhyperparameter)(r_, xhat_, d_, A)
    model = multi_frozenlayer_model(inputs=(y, H), outputs=xhat_)
    return model

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
    SNR = 30
    noise_var = Nt/Nu * tf.pow(10., -SNR / 10.)

    #训练集大小
    train_size = 3030
    #测试集大小
    test_size = 1
    #网络层数
    layersNum = 16
    MAX_EPOCHS = 1
    BATCH_SIZE = 1  # mini-batch set size
    N_TRAIN = 10 ** 5  # training set size
    N_ITER = N_TRAIN // BATCH_SIZE  # number of iterations at each epoch
    TOL = 10 ** -4  # reference value used for convergence
    N_ITER_NO_CHANGE = 10  # reference value used for aborting training
    INIT_ETA = 5e-5  # initial learning rate
    ##########################PART.2 生成网络########################
    #生成网络
    model = NNet(Nu = Nu,Nt = Nt,layersNum = layersNum,SNR=SNR,ifhyperparameter=True)
    ##########################PART.3 训练网络########################

    history = model.multiple_Frozen_compile_fit(  # 使用model.fit()方法来执行训练过程，
        train_size=train_size,
        each_layer_batchsize=BATCH_SIZE,  #
        eachlayer_epochs=MAX_EPOCHS,
        validation_split=0.01,
        validation_freq=1,  # 测试的间隔次数为20
    )
    model.save('model/my_model.h5')
    # 重新加载模型
    #model = models.load_model('model/my_model.h5')
    # print(model.trainable_variables)
    file = open('./weights.txt', 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()
    #######################PART.4 展示各种曲线   ######################
    # 显示训练集和验证集的acc和loss曲线
    acc = history[-1].history['detect_metric']
    val_acc = history[-1].history['detect_metric']
    loss = history[-1].history['loss']
    val_loss = history[-1].history['val_loss']

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