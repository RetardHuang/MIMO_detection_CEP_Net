#########################
#鉴于WeiYi的框架比较复杂，令我欲仙欲死两周时间，我决定重新搭一个框架
#这个框架基于先进的tensorflow2.0 舍弃了很多原来有的语法
#e.g. placeholder 等
#本文描述的是如何生成信道模型。
#我们这里采用y = Hx+n的经典模型

import numpy as np
import math

def npcplx_FOR_X(height,width):
    realPart = 1 - 2*np.random.binomial(size=height*width, n=1, p=0.5).reshape((height,width))
    imagPart = 1 - 2*np.random.binomial(size=height*width, n=1, p=0.5).reshape((height,width))
    return realPart + 1j * imagPart

def npcplx_FOR_N(height,width,noise_var):
    realPart = np.random.normal(size=(height,width), scale=noise_var).astype(np.float32)
    imagPart = np.random.normal(size=(height,width), scale=noise_var).astype(np.float32)
    return realPart + 1j * imagPart
class Channel_generator():
    def __init__(self,Nu,Nt,L_mu=8,noise_var = 0.1):
        self.aChannelMatrix = np.zeros(shape=(Nu, Nt)).astype(np.complex64);
        self.Nu = Nu
        self.Nt = Nt
        self.L_mu = L_mu
        self.noise_var = noise_var
        self.changeH()
        # INITIALIZING CHANNEL
    def changeH(self):
        self.aChannelMatrix = np.random.normal(size=(self.Nu, self.Nt), scale=1.0 / math.sqrt(self.Nu * 2)).astype(np.float32) + 1j * np.random.normal(
            size=(self.Nu, self.Nt), scale=1.0 / math.sqrt(self.Nu * 2)).astype(np.float32)
        self.aChannelMatrixHermit = np.conjugate(np.transpose(self.aChannelMatrix))
    def output(self,setnum = 10,ifreal =True):
        ###############################
        #这里的输入是[x_1 x_2 x_3....x_setnum]
        #这里的输出是[y_1 y_2 y_3....y_setnum]
        #其中x_i,y_i都是纵向量！每一列都是一个样本
        #最后 aChannelMatirx 就是信道矩阵H
        ###############################
        trueX = npcplx_FOR_X(self.Nt, setnum)
        truenoise = npcplx_FOR_N(self.Nu, setnum,noise_var=self.noise_var)
        trueY = np.matmul(self.aChannelMatrix, trueX) + truenoise

        if ifreal:
            Youtput = np.transpose(np.vstack((np.real(trueY),np.imag(trueY))))
            ##
            upH = np.hstack((np.real(self.aChannelMatrix),-np.imag(self.aChannelMatrix)))
            doH = np.hstack((np.imag(self.aChannelMatrix),np.real(self.aChannelMatrix)))
            H = np.vstack((upH,doH))
            ##
            upH = np.hstack((np.real(self.aChannelMatrixHermit),-np.imag(self.aChannelMatrixHermit)))
            doH = np.hstack((np.imag(self.aChannelMatrixHermit),np.real(self.aChannelMatrixHermit)))
            HHermit = np.vstack((upH,doH))

            Xoutput = np.transpose(np.vstack((np.real(trueX),np.imag(trueX))))
            return Xoutput,H,HHermit,Youtput
        else:
            Youtput = np.transpose(trueY)
            Xoutput = np.transpose(trueX)
            return Xoutput,self.aChannelMatrix,self.aChannelMatrixHermit,Youtput
    def multipleoutput(self,setnum = 10,ifreal=True,ifchangeChannel =True):
        XCube = np.empty([setnum,2*self.Nt])
        YCube = np.empty([setnum,2*self.Nu])
        HCube = np.empty([setnum,2*self.Nu,2*self.Nt])
        HHCube = np.empty([setnum,2*self.Nt,2*self.Nu])
        for i in range(0,setnum):
            if ifchangeChannel:
                self.changeH()
            Xoutput,H,HH,Youtput = self.output(setnum=1, ifreal=ifreal)
            XCube[i,:] = Xoutput
            HCube[i,:,:] = H
            HHCube[i,:,:] = HH
            YCube[i,:] = Youtput
        return XCube,HCube,HHCube,YCube
if __name__ == '__main__':
    #如下是使用说明
    channel = Channel_generator(5,3,8)
    channel.changeH()
    print(channel.output(ifreal=False,setnum=1))
    channel.changeH()
    #print(channel.output())
    shit = channel.multipleoutput(30)
    print(shit)
