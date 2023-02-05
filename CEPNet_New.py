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
from tensorflow.math import log
from tensorflow.math import reduce_all
import sys
import numpy as np
class FirstNlayer(keras.layers.Layer):#第一层的单层结构
    def __init__(self, Nu,Nt, SNR,name='First_Layer'):  # block_list表示每个block有几个卷积层
        super().__init__()
        self.Nu = Nu
        self.Nt = Nt
        self.noise_var = self.Nt / self.Nu * tf.pow(10., -SNR / 10.)
    def call(self,y,H,HH):
        A = matmulshit(HH, H) + self.noise_var * tf.eye(2 * self.Nt)
        b = matmulshit(HH, y)
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
            self.alpha_ = self.add_weight(shape=(2*self.Nt, 1),initializer=keras.initializers.Zeros(),trainable=True,name="alpha")
            self.beta_ = self.add_weight(shape=(2*self.Nt, 1), initializer=keras.initializers.Zeros(),trainable=True,name="beta")
        else:#如果是正常参数 则每层权重对应一个 标量
            self.alpha_ = self.add_weight(shape=(1, ), initializer=keras.initializers.Zeros(), trainable=True,name="alpha")
            self.beta_ = self.add_weight(shape=(1, ),initializer=keras.initializers.Zeros(), trainable=True,name="beta")
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
def BER(y_true, y_pred):
    #tf.print("y_true:", y_true, summarize=-1,output_stream=sys.stdout)
    #tf.print("y_pred:", y_pred, summarize=-1,output_stream=sys.stdout)
    return 10*log(tf.reduce_mean(tf.abs(y_true-tf.sign(y_pred))/2))
#我在这里定义一个子类，这个子类的方法中间就是冻结层的训练方法
#为什么不在这个子类里面直接定义网络结构呢，请看我下一段话！
class multi_frozenlayer_model(Model):
    #我自己写的API就是一坨屎，没办法keras本身就是一坨屎，错的不是我，错的是世界
    #你他妈见到过构造函数是在类外边的吗？我他妈没见过
    #如果你把构造函数写在里面等着报bug吧！
    #把构造函数写在里面产生的bug包括两大块：1.矩阵乘法维度不一致；2.其余错误，好吧 第一块把我搞死了
    #怎么办 老子写不完论文了，现在还能用tf1的框架吗？？？？
    def setup_shit(self,Nt,Nu,SNR,L_mu):
        self.Nt = Nt
        self.Nu = Nu
        self.SNR = SNR
        self.L_mu = L_mu
    def setup_data_optimizer(self):
        # 生成信道模型
        noise_var = self.Nt / self.Nu * tf.pow(10., -self.SNR / 10.)
        self.channel = Channel_generator(Nu=self.Nu, Nt=self.Nt, L_mu=self.L_mu, noise_var=noise_var)
        # 训练器的学习率衰减
        #三种Optimizer

        checkpoint_save_path = "./checkpoint/cep8.ckpt"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            self.load_weights(checkpoint_save_path)

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
        save_weights_only=True,
        save_best_only=True)
        self.tensorboard_callback = TensorBoard(log_dir=r"./mytensotboard")
        self.stopcallback = tf.keras.callbacks.EarlyStopping(monitor="loss", mode = "min",min_delta=1e-3,verbose = 1,patience = 3)
        self.summary()
        pass
    def multiple_Frozen_compile_fit(self,train_size,
                                    each_layer_batchsize,
                                    eachlayer_epochs,
                                    validation_split,
                                    validation_freq
                                    ):
        self.setup_data_optimizer()
        # 全部冻结
        for layer in self.layers:
            layer.trainable = False
        # 依次开启层，然后训练
        history=[]
        number = 0
        for layer in self.layers :
            if len(layer.non_trainable_variables)!=0:
                number = number + 1
                #
                print('Now, training layer:',layer.name)
                #打开该层
                layer.trainable = True
                # 生成训练集
                XCube, HCube,HHCube, YCube = self.channel.multipleoutput(setnum=train_size, ifreal=True, ifchangeChannel=True)
                # 编译网络，也是载, 入优化器和损失函数的地方
                #大型
                big_optimizer_adam = keras.optimizers.Adam(learning_rate=1e-3)
                #小型
                small_optimizer_adam = keras.optimizers.Adam(learning_rate=5e-4)

                if number%2 ==1:
                    self.compile(optimizer=big_optimizer_adam,
                                 loss=loss_norm_nmse,
                                 metrics=[BER]
                                 )
                else:
                    self.compile(optimizer=small_optimizer_adam,
                                 loss=loss_norm_nmse,
                                 metrics=[BER]
                                 )
                # 断点存储位置
                history.append(self.fit(  # 使用model.fit()方法来执行训练过程，
                    x= [YCube,HCube,HHCube], y = XCube,  # 告知训练集的输入以及标签，
                    batch_size=each_layer_batchsize,  # 每一批batch的大小为32，
                    epochs=eachlayer_epochs,
                    validation_split=validation_split,  # 从测试集中划分80%给训练集
                    validation_freq=validation_freq,  # 测试的间隔次数为20
                    callbacks=self.tensorboard_callback
                )
                )
                #该层关闭
                layer.trainable = False
            else:
                pass
        return history
    #下面的这个方法是采用的Wei Yi论文里的方法，即一层一层训练，也就是只启用一层，其余层冻结
    def normal_compile_fit(self,train_size,
                                    batchsize,
                                    epochs,
                                    validation_split,
                                    validation_freq
                                    ):
        #指数衰减型损失函数
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_steps=train_size,decay_rate=0.9)
        self.optimizer_adm = keras.optimizers.Adam(learning_rate = lr_schedule)  # instantiate the solver
        self.setup_data_optimizer()
        self.compile(optimizer=self.optimizer_adm,
                     loss=loss_norm_nmse,
                     metrics=[BER]
                     )
        XCube, HCube, HHCube, YCube = self.channel.multipleoutput(setnum=train_size, ifreal=True, ifchangeChannel=True)
        history = self.fit(  # 使用model.fit()方法来执行训练过程，
            x=[YCube, HCube, HHCube], y=XCube,  # 告知训练集的输入以及标签，
            batch_size=batchsize,  # 每一批batch的大小为32，
            epochs=epochs,
            validation_split=validation_split,  # 从测试集中划分80%给训练集
            validation_freq=validation_freq,  # 测试的间隔次数为20
            callbacks=[self.tensorboard_callback,self.cp_callback,self.stopcallback]
        )
        return history

    def multiple_predict(self,test_size,SNR,iflogSER = True):
        noise_var = self.Nt / self.Nu * tf.pow(10., -SNR / 10.)
        predict_channel=Channel_generator(self.Nu,self.Nt,L_mu=self.L_mu,noise_var = noise_var)
        XCube, HCube, HHCube, YCube = predict_channel.multipleoutput(setnum=test_size, ifreal=True, ifchangeChannel=True)
        predictions = self.predict([YCube,HCube,HHCube], batch_size=1)
        predicted_hard = tf.squeeze(tf.sign(predictions),axis = 2)#为了删除第三个长度为1的维，用squeeze
        Truth = XCube == predicted_hard
        shit = reduce_all(Truth,axis=1)
        num_true = tf.reduce_sum(tf.cast(shit, tf.int32))
        SER = (test_size - num_true)/test_size
        if iflogSER:
            if SER ==0:
                return -math.inf
            else:
                return math.log10(SER)
        else :
            return SER
    def get_config(self):
        config = super().get_config()
        return config


#我在这里吐槽下，Model的子类构建方法弄得跟屎一样！你根本无法采用批学习
#如果你在modelclass里面定义网络结构，会出现 权重无法点乘的傻逼问题！！！
#而且，乘法也要注意，如果时矩阵乘向量，必须要采用linalg.matvec这个函数
#如果还是你不知道怎么回事的话，你就运行一下CEPNet_modelclass.py吧
#####真尼玛傻逼！！！！#####
#keras，我真尼玛后悔用，早知道用tf1的框架了！
#所以这里我们在类外部构建网络！
def NNet(Nu,Nt, SNR,layersNum=20,ifhyperparameter = False):#多层结构
    H = Input(shape=(2*Nu,2*Nt), name='H')
    HH = Input(shape=(2*Nt,2*Nu), name='HH')
    y = Input(shape=(2*Nu,1), name='y')
    r_, xhat_, d_, A = FirstNlayer(Nu=Nu, Nt=Nt, SNR=SNR)(y,H,HH)
    for layer_id in range(1, layersNum):  # 第几个卷积层
        r_, xhat_, d_, A = Nlayer(Nu=Nu, Nt=Nt, ifhyperParameter=ifhyperparameter)(r_, xhat_, d_, A)
    model = multi_frozenlayer_model(inputs=(y, H,HH), outputs=xhat_)
    model.setup_shit(Nt,Nu,SNR,L_mu= 8)
    model.setup_data_optimizer()
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

    #训练集大小
    train_size = 40400
    #测试集大小
    test_size = 1
    #网络层数
    layersNum = 30
    MAX_EPOCHS = 5
    BATCH_SIZE = 5  # mini-batch set size
    N_TRAIN = 10 ** 5  # training set size
    N_ITER = N_TRAIN // BATCH_SIZE  # number of iterations at each epoch
    TOL = 10 ** -4  # reference value used for convergence
    N_ITER_NO_CHANGE = 10  # reference value used for aborting training
    INIT_ETA = 5e-5  # initial learning rate
    ##########################PART.2 生成网络########################
    #生成网络
    model = NNet(Nu = Nu,Nt = Nt,layersNum = layersNum,SNR=SNR,ifhyperparameter=True)
    ##########################PART.3 训练网络########################

    history = model.normal_compile_fit(  # 使用model.fit()方法来执行训练过程，
        train_size=train_size,
        each_layer_batchsize=BATCH_SIZE,  #
        eachlayer_epochs=MAX_EPOCHS,
        validation_split=0.01,
        validation_freq=1,  # 测试的间隔次数为20
    )
    #savepath = tf.train.latest_checkpoint('/content/training_2')
    #model.save(savepath)
    model.save_weights("my_model.h5")
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
    acc = history.history['BER']
    val_acc = history.history['BER']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='BER')
    plt.plot(val_acc, label='BER')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()