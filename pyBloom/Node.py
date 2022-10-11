'''
    node为整体树结构的节点，包含左右两个bloom filter,以及向子节点的指针
    MlModel为底层的模型实现
'''

#组件包
from Bloomfilter.BloomFilter import BloomFilter
#科学包
#from sklearn.linear_model import LinearRegression
from model_lib import LinearRegression
import numpy as np
from scipy.stats import norm
#基础包
import math
from memory_profiler import profile

class node:
    '''定义自己的左右两个bloom filter以及左右子节点'''
    def __init__(self,level,lefcap,leferr,rigcap,rigerr):
        self.level=level
        self.leftfilter=BloomFilter(lefcap,leferr)
        self.rightfilter=BloomFilter(rigcap,rigerr)
        self.lefchld=None
        self.rigchld=None
    '''提供查询数字是否存在的服务'''
    def Contains(self,num):
        # bias=num>>((self.level-1)<<4)
        bias=num>>(32-self.level)
        flag=bias & 1
        '''第一层取前16bits进入bloom filter,第二层取全部32bits'''
        # mask = ~((-1)<<(16+((self.level-1)<<4)))
        # temp= num & mask
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
        self.model_flag = model_flag #0为线性模型，1为DNN
        if self.model_flag==0:
            self.model = LinearRegression()
        else:
            self.model = None
        self.data = None
        self.data_size = None
        self.error_bound = []
        self.train(data)

    def train(self,x):##nparray形式传入
        #print(x)
        if self.model_flag == 0:
            self.data = x
            self.data_size = N = len(x)
            label = np.arange(N)
            x = x.reshape(-1,1)
            self.model.fit(x,label)
            min_err = max_err = average_error = 0
            for i in range(N):
                pos = int( self.model.predict([ x[i] ]) )#x[i]已经是列表了
                err = i-pos
                average_error += abs(err) /N
                if err < min_err:
                    min_err = math.floor(err)
                elif err > max_err:
                    max_err = math.ceil(err)
            self.error_bound = [min_err, max_err]


    # def cdf(x):
    #     loc = x.mean()
    #     scale = x.std()
    #     n = x.size
    #     pos = norm.cdf(x, loc, scale) * n
    #     return pos

    #@profile
    def search(self,key):
        #print("searching for ",key)
        pos = int(self.model.predict([[key]]))
        if pos < 0:
            lbound = pos = 0
        if pos > self.data_size - 1:
            ubound = pos = self.data_size - 1

        if self.data[pos] == key :
            return pos
        lbound = pos+self.error_bound[0]
        ubound = pos+self.error_bound[1]
        #print(lbound,ubound)
        #for i in range(lbound,ubound):
        # 检查预测的位置是否超过范围
        if pos < 0:
            lbound = pos = 0
        if pos > self.data_size - 1:
            ubound = pos = self.data_size - 1

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












