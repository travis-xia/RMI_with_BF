

import numpy as np
import matplotlib.pyplot as plt
import random
def GenerateData(start,end,num):#生成加入高斯干扰的曲线
    X=np.linspace(start,end,num)
    Y=np.sin(X)
    miu=0
    sigma=0.12
    for i in range(X.size):
        X[i]+=random.gauss(miu,sigma)
        Y[i]+=random.gauss(miu,sigma)
    # print(X)
    # print(Y)
    return (X,Y)
def CalculateGradient(dataset,coefficientset,labelset,datasize):#计算梯度
    result =(1/datasize)*np.dot(dataset.T, (np.dot(dataset, coefficientset) - labelset))
    return result
def lossfunc(dataset,coefficientset,labelset,datasize):
    result=0.5*(1/datasize)*np.dot((labelset-np.dot(dataset, coefficientset)).T,(labelset-np.dot(dataset, coefficientset)))
    return result
def GradientDown(dataset,coefficientset,labelset):#梯度下降算法
    length=0.01#步长
    loss0=0
    loss1=lossfunc(dataset,coefficientset,labelset,40)
    print('loss为:')
    print(loss1)
    while abs(loss1 - loss0) >1e-8:#当两次loss的差值接近时认为迭代完成
        gradient=CalculateGradient(dataset,coefficientset,labelset,40)#计算梯度
        print('梯度为')
        print(gradient)
        coefficientset=coefficientset-length*gradient#梯度下降
        loss0=loss1
        print('原loss为：')
        print(loss0)
        loss1=lossfunc(dataset,coefficientset,labelset,40)
        print('现loss为：')
        print(loss1)
        if np.all((loss1 - loss0) > 0):#当loss不降反升的时候减小步长
            length *= 0.1
            print('现在步长更新为：')
            print(length)
    print('最终系数为')
    print(coefficientset)
    return coefficientset#返回最终得到的系数向量
def GenerateX(X,times):
    result=[]
    for i in range (0,X.size):
        for j in range(times):
            result.append(X[i]**(times-j-1))
    return result
def Generatexishu(times):
    result=[]
    for i in range(times):
        result.append(0.1)
    return result
def ConjugateGradient(dataset,labelset,lamda):
    # A = np.dot(dataset.T, dataset) + lamda * np.eye(dataset.shape[1])  #n+1 * n+1
    A = np.dot(dataset.T, dataset)
    b = np.dot(dataset.T, labelset)  # n+1 * 1
    w = np.zeros((dataset.shape[1], 1))  # 初始化w为 n+1 * 1 的零阵
    r = b
    p = b
    while True:
        if ((r.T).dot(r).tolist()[0][0]) <=1e-8:
            break
        norm_2 =(r.T).dot(r).tolist()[0][0]#该点梯度的平方和
        print('误差为')
        print(norm_2)
        a = norm_2 / np.dot(p.T, A).dot(p) #计算出步长
        a=a.tolist()[0][0]
        w = w + a * p
        r = r - (a * A).dot(p)#计算出新的梯度
        b = np.dot(r.T, r).tolist()[0][0] / norm_2#替换系数公式
        p = r + b * p
    return w
def ConjugateGradient2(dataset,labelset,lamda):
    A = np.dot(dataset.T, dataset) + lamda * np.eye(dataset.shape[1])  #n+1 * n+1
    b = np.dot(dataset.T, labelset)  # n+1 * 1
    w = np.zeros((dataset.shape[1], 1))  # 初始化w为 n+1 * 1 的零阵
    r = b
    p = b
    while True:
        if ((r.T).dot(r).tolist()[0][0]) <=1e-8:
            break
        norm_2 =(r.T).dot(r).tolist()[0][0]#该点梯度的平方和
        print('误差为')
        print(norm_2)
        a = norm_2 / np.dot(p.T, A).dot(p) #计算出步长
        a=a.tolist()[0][0]
        w = w + a * p
        r = r - (a * A).dot(p)#计算出新的梯度
        b = np.dot(r.T, r).tolist()[0][0] / norm_2#替换系数公式
        p = r + b * p#计算出新方向
    return w
def gradientregression(X,Y):
    K = X
    D = Y
    dataset = []
    dataset=GenerateX(X,5)
    print(dataset)
    dataset=np.mat(np.array(dataset).reshape(10,5))
    # print(dataset)
    Y=np.mat(Y).reshape(10,1)
    # print(Y)
    xishu=[]
    xishu.append(0)
    xishu.append(0.1)
    xishu.append(0)
    xishu.append(0)
    xishu.append(0)
    w=np.mat(xishu).reshape(5,1)
    # print(w)
    dataset2=GenerateX(X,9)
    dataset2= np.mat(np.array(dataset2).reshape(10, 9))
    xishu2=Generatexishu(9)
    w2 = np.mat(xishu2).reshape(9, 1)
    newset=GradientDown(dataset,w,Y)
    print('系数为:')
    print(newset)
    k1 = float(newset[0][0])
    k2 = float(newset[1][0])
    k3 = float(newset[2][0])
    k4 = float(newset[3][0])
    k5=float(newset[4][0])
    xzhou = np.arange(0, 2 * np.pi, 0.001)
    yzhou = k1 * (xzhou ** 4) + k2 * (xzhou ** 3) + k3 * (xzhou**2)+k4*xzhou + k5
    plt.plot(xzhou, yzhou,color='r')
    plt.plot(K, D, linestyle='', marker='.')
    # plt.title=('4times/gradient/without regular')
    plt.show()
def Plusregression(X,Y):
    dataset = GenerateX(X, 10)
    dataset = np.mat(np.array(dataset).reshape(10, 10))
    labelset= np.mat(Y).reshape(10, 1)
    print('y值为:')
    print(np.dot(labelset.T,labelset))
    w=ConjugateGradient(dataset,labelset,1)
    j1 = float(w[0][0])
    j2 = float(w[1][0])
    j3 = float(w[2][0])
    j4 = float(w[3][0])
    j5 = float(w[4][0])
    j6 = float(w[5][0])
    j7 = float(w[6][0])
    j8 = float(w[7][0])
    j9 = float(w[8][0])
    j10 = float(w[9][0])
    xzhou = np.arange(0, 2 * np.pi, 0.01)
    yzhou2 = j1 * xzhou ** 9 + j2* xzhou ** 8 + j3 * xzhou ** 7 + j4* xzhou ** 6 + j5 * xzhou ** 5 + j6* xzhou ** 4 + j7 * xzhou ** 3 + j8 * xzhou ** 2 + j9 * xzhou + j10
    plt.plot(xzhou, yzhou2, color='b')
    plt.plot(X, Y, linestyle='', marker='*', color='g')
    plt.title('conjugategradient/9times/without regular')
    plt.show()
def Plusregression2(X,Y):
    dataset = GenerateX(X, 5)
    dataset = np.mat(np.array(dataset).reshape(10, 5))
    labelset= np.mat(Y).reshape(10, 1)
    print('y值为:')
    print(np.dot(labelset.T,labelset))
    w=ConjugateGradient(dataset,labelset,1)
    j1 = float(w[0][0])
    j2 = float(w[1][0])
    j3 = float(w[2][0])
    j4 = float(w[3][0])
    j5 = float(w[4][0])
    xzhou = np.arange(0, 2 * np.pi, 0.01)
    yzhou2 = j1 * xzhou ** 4 + j2 * xzhou ** 3 + j3 * xzhou ** 2 + j4 * xzhou ** 1 + j5
    plt.plot(xzhou, yzhou2, color='b')
    plt.plot(X, Y, linestyle='', marker='*', color='g')
    plt.title('conjugategradient/4times/without regular')
    plt.show()

def Plusregression3(X, Y):
    dataset = GenerateX(X, 10)
    dataset = np.mat(np.array(dataset).reshape(10, 10))
    labelset = np.mat(Y).reshape(10, 1)
    print('y值为:')
    print(np.dot(labelset.T, labelset))
    w = ConjugateGradient2(dataset, labelset, 1)
    j1 = float(w[0][0])
    j2 = float(w[1][0])
    j3 = float(w[2][0])
    j4 = float(w[3][0])
    j5 = float(w[4][0])
    j6 = float(w[5][0])
    j7 = float(w[6][0])
    j8 = float(w[7][0])
    j9 = float(w[8][0])
    j10 = float(w[9][0])
    xzhou = np.arange(0, 2 * np.pi, 0.01)
    yzhou2 = j1 * xzhou ** 9 + j2 * xzhou ** 8 + j3 * xzhou ** 7 + j4 * xzhou ** 6 + j5 * xzhou ** 5 + j6 * xzhou ** 4 + j7 * xzhou ** 3 + j8 * xzhou ** 2 + j9 * xzhou + j10
    plt.plot(xzhou, yzhou2, color='b')
    plt.plot(X, Y, linestyle='', marker='*', color='g')
    plt.title('conjugategradient/9times/with regular')
    plt.show()
def Plusregression5(X, Y):
    dataset = GenerateX(X, 10)
    dataset = np.mat(np.array(dataset).reshape(10, 10))
    labelset = np.mat(Y).reshape(10, 1)
    print('y值为:')
    print(np.dot(labelset.T, labelset))
    w = ConjugateGradient2(dataset, labelset, 100000)
    j1 = float(w[0][0])
    j2 = float(w[1][0])
    j3 = float(w[2][0])
    j4 = float(w[3][0])
    j5 = float(w[4][0])
    j6 = float(w[5][0])
    j7 = float(w[6][0])
    j8 = float(w[7][0])
    j9 = float(w[8][0])
    j10 = float(w[9][0])
    xzhou = np.arange(0, 2 * np.pi, 0.01)
    yzhou2 = j1 * xzhou ** 9 + j2 * xzhou ** 8 + j3 * xzhou ** 7 + j4 * xzhou ** 6 + j5 * xzhou ** 5 + j6 * xzhou ** 4 + j7 * xzhou ** 3 + j8 * xzhou ** 2 + j9 * xzhou + j10
    plt.plot(xzhou, yzhou2, color='b')
    plt.plot(X, Y, linestyle='', marker='*', color='g')
    plt.title('conjugategradient/9times/with regular/punish lot')
    plt.show()
def simpleregression(X,Y):
    K=X
    D=Y
    dataset=[]
    # for i in range(0,X.size):
    #     dataset.append(X[i]**9)
    #     dataset.append(X[i]**8)
    #     dataset.append(X[i] ** 7)
    #     dataset.append(X[i] ** 6)
    #     dataset.append(X[i] ** 5)
    #     dataset.append(X[i] ** 4)
    #     dataset.append(X[i]*X[i]*X[i])
    #     dataset.append(X[i]*X[i])
    #     dataset.append(X[i])
    #     dataset.append(1)
    dataset=GenerateX(X,10)
    print(len(dataset))
    #dataset=np.mat(np.array(dataset).reshape(10,4))
    dataset = np.mat(np.array(dataset).reshape(10, 10))
    Y=np.mat(Y).reshape(10,1)
    print(dataset)
    #print(Y)
    w=(np.linalg.inv(np.dot(np.transpose(dataset),dataset))).dot(np.transpose(dataset)).dot(Y)#直接通过表达式运算求解系数
    w2=(np.linalg.inv(np.dot(np.transpose(dataset),dataset)+np.eye(10))).dot(np.transpose(dataset)).dot(Y)
    print('系数为:')
    print(w)
    print(w2)
    k1=float(w[0][0])
    k2 = float(w[1][0])
    k3 = float(w[2][0])
    k4 = float(w[3][0])
    k5 = float(w[4][0])
    k6 = float(w[5][0])
    k7 = float(w[6][0])
    k8 = float(w[7][0])
    k9=float(w[8][0])
    k10=float(w[9][0])
    j1 = float(w2[0][0])
    j2 = float(w2[1][0])
    j3 = float(w2[2][0])
    j4 = float(w2[3][0])
    j5 = float(w2[4][0])
    j6 = float(w2[5][0])
    j7 = float(w2[6][0])
    j8 = float(w2[7][0])
    j9 = float(w2[8][0])
    j10 = float(w2[9][0])
    xzhou=np.arange(0,2*np.pi,0.01)
    #yzhou=k1*xzhou**3+k2*xzhou**2+k3*xzhou+k4
    yzhou =k1*xzhou**9+k2*xzhou**8+k3 *xzhou**7+k4*xzhou**6+k5*xzhou**5+k6*xzhou**4+k7*xzhou**3+k8*xzhou**2+k9*xzhou+k10
    yzhou2 = j1 * xzhou ** 9 + j2 * xzhou ** 8 + j3 * xzhou ** 7 + j4 * xzhou ** 6 + j5 * xzhou ** 5 + j6 * xzhou ** 4 + j7 * xzhou ** 3 + j8 * xzhou ** 2 + j9 * xzhou + j10
    y2=np.sin(xzhou)
    plt.plot(xzhou,yzhou,color='r')
    # plt.plot(xzhou,y2,':')
    plt.plot(xzhou,yzhou2,color='b')
    plt.plot(K, D, linestyle='', marker='*',color='g')
    plt.title("9times with regular without regular analyses")
    plt.show()
def Plusregression4(X,Y):
    dataset = GenerateX(X, 10)
    dataset = np.mat(np.array(dataset).reshape(20, 10))
    labelset= np.mat(Y).reshape(20, 1)
    print('y值为:')
    print(np.dot(labelset.T,labelset))
    w=ConjugateGradient(dataset,labelset,1)
    j1 = float(w[0][0])
    j2 = float(w[1][0])
    j3 = float(w[2][0])
    j4 = float(w[3][0])
    j5 = float(w[4][0])
    j6 = float(w[5][0])
    j7 = float(w[6][0])
    j8 = float(w[7][0])
    j9 = float(w[8][0])
    j10 = float(w[9][0])
    xzhou = np.arange(0, 2 * np.pi, 0.01)
    yzhou2 = j1 * xzhou ** 9 + j2 * xzhou ** 8 + j3 * xzhou ** 7 + j4 * xzhou ** 6 + j5 * xzhou ** 5 + j6 * xzhou ** 4 + j7 * xzhou ** 3 + j8 * xzhou ** 2 + j9 * xzhou + j10
    plt.plot(xzhou, yzhou2, color='b')
    plt.plot(X, Y, linestyle='', marker='*', color='g')
    plt.title('conjugategradient/9times/without regular')
    plt.show()
(X,Y)=GenerateData(0,2*np.pi,10)
# gradientregression(X,Y)
Plusregression2(X,Y)
simpleregression(X,Y)
Plusregression(X,Y)
Plusregression3(X,Y)
Plusregression5(X,Y)
(W,Z)=GenerateData(0,2*np.pi,20)
Plusregression4(W,Z)