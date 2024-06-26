# -*- coding: UTF-8 -*-
import math
import time

import numpy as np
from scipy.stats import norm
from model_lib import LinearRegression
from sklearn import preprocessing
from memory_profiler import profile
def cdf(x):
    if type(x) == np.ndarray:
        loc = x.mean()
        scale = x.std()
        n = x.size
        pos = norm.cdf(x, loc, scale) * n
        return pos
    else:
        print("Wrong Type!~")
        exit(-1)

class RmiModel:
    @profile(precision=4)
    def __init__(self):
        self.average_error = 0
        self.num_level = [1,2,4]
        self.model_index = []
        for i in self.num_level:
            temp = []
            for j in range(i):
                temp_model = LinearRegression()
                temp.append(temp_model)
            self.model_index.append(temp)
        self.data = np.array([])
        self.mean = None  # 存储均值
        self.std = None  # 存储差标准差
        self.mean_ = []
        self.std_ = []
        self.sub_size = []
        self.error_bound = []

    def train(self,data):
        self.data = data
        self.mean = data.mean()
        self.std = data.std()
        data = np.array(data)
        self.data_size = len(data)
        label  = np.arange(self.data_size)
        sub_data = []
        sub_label = []
        cdf_label = cdf(data)
        norm_data = (data - np.mean(data)) / np.std(data)

        self.model_index[0][0].fit(norm_data,cdf_label)

        m = self.num_level[1]
        sub_data = [[] for _ in range(m)]
        sub_y = [[] for _ in range(m)]
        for i in range(self.data_size):
            mm = int(self.model_index[0][0].predict([[norm_data[i]]]) * m / self.data_size)
            if mm < 0:
                mm = 0
            elif mm > m - 1:
                mm = m - 1
            sub_data[mm].append(data[i])
            sub_y[mm].append(i)
        self.sub_size = [len(sub_data[0]),len(sub_data[1])]
        for i in range(m):
            sub_norm_data = (sub_data[i] - np.mean(sub_data[i])) / np.std(sub_data[i])
            self.model_index[1][i].fit(np.array(sub_norm_data),cdf(np.array(sub_data[i])))

        m = self.num_level[2]
        subsub_data = [[] for _ in range(m)]
        subsub_y = [[] for _ in range(m)]
        for i in range(self.num_level[1]):
            mean = np.mean(sub_data[i])
            self.mean_.append(mean)
            std = np.std(sub_data[i])
            self.std_.append(std)
            for j in range(len(sub_data[i])):
                mm = int(self.model_index[1][i].predict([[(sub_data[i][j] - mean) / std]]) *2 /len(sub_data[i]) )
                if mm < 0:
                    mm = 0
                elif mm > 2 - 1:
                    mm = 2 - 1
                mm = mm+2*i
                # print(mm)
                subsub_data[mm].append(sub_data[i][j])
                subsub_y[mm].append(sub_y[i][j])
        for i in range(m):
            if len(subsub_data[i])<=0:
                continue
            self.model_index[2][i].fit(np.array(subsub_data[i]),np.array(subsub_y[i]))


        min_err = [0,0,0,0]
        max_err = [0,0,0,0]
        for t in range(data.size):
            if t % 100000 == 0:
                print("    ", t)
            ppos, mmm = self.predict(data[t])
            # print(data[t],t,ppos,mmm)
            err = t - ppos
            self.average_error += abs(err)
            if err < min_err[mmm]:
                min_err[mmm] = math.floor(err)
            elif err > max_err[mmm]:
                max_err[mmm] = math.ceil(err)
        self.average_error /= data.size
        print("average error:", self.average_error * 1.0 / self.data_size)
        for i in range(4):
            self.error_bound.append([min_err[i], max_err[i]])
        print(self.error_bound)

    @profile(precision=4)
    def search(self,key):
        pos, model_index = self.predict(key)
        lp = pos + self.error_bound[model_index][0]
        rp = pos + self.error_bound[model_index][1]
        # 检查预测的位置是否超过范围
        if pos < 0:
            lp = pos = 0
        if pos > self.data_size - 1:
            rp = pos = self.data_size - 1
        if self.data[pos] == key:
            return pos
        if lp < 0:
            lp = 0
        if rp > self.data_size - 1:
            rp = self.data_size - 1
        while lp <= rp:
            if self.data[pos] == key:
                return pos
            elif self.data[pos] > key:
                rp = pos - 1
            elif self.data[pos] < key:
                lp = pos + 1
            pos = int((lp + rp) / 2)
        return False
    def predict(self,key):
        mm = int(self.model_index[0][0].predict([[(key - self.mean) / self.std]]) * self.num_level[1] / self.data_size)
        if mm < 0:
            mm = 0
        elif mm > self.num_level[1] - 1:
            mm = self.num_level[1] - 1

        mmm = int(self.model_index[1][mm].predict([[(key - self.mean_[mm]) / self.std_[mm]]]) * 2 / self.sub_size[mm])
        if mmm < 0:
            mmm = 0
        elif mmm > 2 - 1:
            mmm = 2 - 1
        mmm = mmm+2*mm
        pos = int(self.model_index[2][mmm].predict([[key]]))
        return pos, mmm





def Test_Simple_main():
    MB =  int(input("请输入数据大小(1-200)MB:"))
    data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    data_size = MB *1000000
    if MB ==200:
        del data
        data_choice = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    else:
        data_choice = np.random.choice(data, data_size)
        del data
    data_choice.sort()
    print('data ok')
    rmi_test = RmiModel()
    print("training begin")
    rmi_test.train(data_choice)


    print("@positive search:")
    data_test = np.random.choice(data_choice, 1)
    for i in data_test:
        rmi_test.search(i)
    print("-------end--------")



if __name__ == '__main__':
    Test_Simple_main()
