import numpy as np
import tensorflow as tf
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from Channel_generator import Channel_generator
from CEPNet import NNet
#设置随机种子
np.random.seed(1)
# 当数组元素比较多的时候，如果输出该数组，那么会出现省略号,这一句让输出变为全部
np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    ########################PART.0 参数配置############################
    #发端天线数量
    Nt = 4
    #收端天线数量
    Nu = 8
    #多径数
    L_mu = 8
    #白噪声方差
    noise_var = 0.1
    #训练集大小
    train_size = 1000
    #测试集大小
    test_size = 50

    #网络层数
    layersNum = 8

    ##########################PART.1 生成测试数据#######################
    #生成信道模型
    channel = Channel_generator(Nu = Nu,Nt = Nt,L_mu = L_mu,noise_var = noise_var)
    #初始化信道矩阵
    #生成训练集
    train_list= channel.multipleoutput(setnum=train_size,ifreal=True)
    #生成测试集
    #(x_test, H ,y_test)= channel.output(setnum=test_size,ifreal=True)

    ##########################PART.2 生成网络########################
    #生成网络
    model = NNet(Nu = Nu,Nt = Nt,layersNum = 8)

    #编译网络，也是载入优化器和损失函数的地方
    model.compile(optimizer='adam',
                  loss='mse'
                  )

    #断点存储位置
    checkpoint_save_path = "./checkpoint/cep8.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    ##########################PART.3 训练网络########################

    history = model.fit(  # 使用model.fit()方法来执行训练过程，
        y_train, x_train,  # 告知训练集的输入以及标签，
        batch_size=1,  # 每一批batch的大小为32，
        epochs=1,  # 迭代次数epochs为500
        validation_split=0.2,  # 从测试集中划分80%给训练集
        validation_freq=20  # 测试的间隔次数为20
    )
    model.summary()

    # print(model.trainable_variables)
    file = open('./weights.txt', 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

    #######################PART.4 展示各种曲线   ######################

    # 显示训练集和验证集的acc和loss曲线
    loss = history.history['loss']
    val_loss = history.history['val_loss']


    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()