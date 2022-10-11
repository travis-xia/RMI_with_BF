# -*- coding: UTF-8 -*-
'''
    node为整体树结构的节点，包含左右两个bloom filter,以及向子节点的指针
    MlModel为底层的模型实现
'''

#组件包
from BloomFilter import BloomFilter
#科学包
#from sklearn.linear_model import LinearRegression
from model_lib import LinearRegression,PolyRegression
import numpy as np
from scipy.stats import norm
#基础包
import math
#from memory_profiler import profile

class node:
    '''定义自己的左右两个bloom filter以及左右子节点'''
    def __init__(self,level,lefcap,leferr,rigcap,rigerr):
        self.level=level
        self.leftfilter=BloomFilter(lefcap,leferr)
        self.rightfilter=BloomFilter(rigcap,rigerr)
        self.lefchld=None
        self.rigchld=None
        # self.BFs = [self.rightfilter,self.leftfilter]
    '''提供查询数字是否存在的服务'''
    def Contains(self,num):
        # bias=num>>((self.level-1)<<4)
        bias=num>>(32-self.level)
        flag=bias & 1
        # mask = ~((-1)<<(16+((self.level-1)<<4)))
        # temp= num & mask
        # return num in self.BFs[flag]
        if flag==1:
            # return temp in self.leftfilter
            return num in self.leftfilter
        else:
            # return temp in self.rightfilter
            return num in self.leftfilter
    '''提供添加元素的服务'''
    def Add(self,num,flag):
        # print("开始添加数字"+str(num))
        if flag==1:
            return self.leftfilter.add(num)
        else:
            return self.rightfilter.add(num)



class MLmodel:
    def __init__(self,data,model_flag=0):
        self.model_flag = model_flag #0为线性模型，1为多项式回归，2为DNN
        if self.model_flag==0:
            self.model = LinearRegression()
        else:
            self.model = PolyRegression()
        self.data = None
        self.data_size = None
        self.error_bound = []
        self.train(data)

    def train(self,x):##nparray形式传入
        #print(x)
        if self.model_flag == 0 or 1:
            self.data = x
            self.data_size = N = len(x)
            label = np.arange(N)
            x = x.reshape(-1,1)
            self.model.fit(x,label)
            min_err = max_err = average_error = 0
            #print(N,int(np.sqrt(N)))
            # randomChoice = np.random.randint(0,N,int(np.sqrt(N)))
            # randomChoice.sort()
            # for i in randomChoice:
            print('this model\'s data',self.data)
            print("    error bound calculating...")
            for i in range(N):
                if i%100000==0 :
                    print("    ",i)
                pos = int( self.model.predict([ x[i] ]) )#x[i]已经是列表了
                err = i-pos
                average_error += abs(err) /N
                if err < min_err:
                    min_err = math.floor(err)
                elif err > max_err:
                    max_err = math.ceil(err)
            print("    error bound :",min_err,max_err)
            print("average error:",1.0*average_error / self.data_size * 100)
            self.error_bound = [min_err, max_err]


    #@profile
    def search(self,key):
        #print("searching for ",key)
        a = self.model.predict([[key]])
        # if np.isnan(a):
        #     return False
        pos = int(a)
        if pos < 0:
            lbound = pos = 0
        if pos > self.data_size - 1:
            ubound = pos = self.data_size - 1
        if self.data[pos] == key :
            return pos
        lbound = pos+self.error_bound[0]
        ubound = pos+self.error_bound[1]
        if lbound < 0:
            lbound = 0
        if ubound > self.data_size - 1:
            ubound = self.data_size - 1
        while lbound <= ubound:
            if self.data[pos] == key:
                #print("successful search:",key,pos)
                return pos
            elif self.data[pos] > key:
                ubound = pos - 1
            elif self.data[pos] < key:
                lbound = pos + 1
            pos = int((lbound + ubound) / 2)
        return False












