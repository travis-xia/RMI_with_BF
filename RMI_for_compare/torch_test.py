# -*- coding: UTF-8 -*-
import math
import time

import numpy as np
import torch
from torch.autograd import Variable as var
from scipy.stats import norm
from model_lib import LinearRegression
from sklearn import preprocessing


def get_data(x, w, b, d):
    c, r = x.shape
    y = (w * x * x + b * x + d) + (0.1 * (2 * np.random.rand(c, r) - 1))
    return y


xs = np.arange(0, 3, 0.01).reshape(-1, 1)
ys = get_data(xs, 1, -2, 3)

xs = torch.Tensor(xs)
ys = torch.Tensor(ys)


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.linear1 = torch.nn.Linear(1, 16)
        self.linear2 = torch.nn.Linear(16, 16)
        self.linear3 = torch.nn.Linear(16, 1)
        self.relu = torch.nn.ReLU()

        self.criterion = torch.nn.MSELoss()
        self.opt = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, inputx):
        y = self.relu(self.linear1(inputx))
        y = self.relu(self.linear2(y))
        y = self.linear3(y)
        return y


# 封装一下
class SecondModel:
    def __init__(self):
        self.E = 500
        self.model = BaseModel()

    def fit(self, inputx, inputy):
        for e in range(self.E):
            y_pre = self.model(inputx)
            loss = self.model.criterion(y_pre, inputy)
            if e % int(self.E // 4) == 0:
                print(e, loss.data)

            # Zero gradients
            self.model.opt.zero_grad()
            # perform backward pass
            loss.backward()
            # update weights
            self.model.opt.step()

    def predict(self, x):
        return self.model(x)


def test_train():
    # print('CUDA GPU index:', torch.cuda.current_device())
    model = BaseModel()
    for e in range(2000):
        y_pre = model(xs)
        loss = model.criterion(y_pre, ys)
        if e % 500 == 0:
            print(e, loss.data)

        # Zero gradients
        model.opt.zero_grad()
        # perform backward pass
        loss.backward()
        # update weights
        model.opt.step()


def cdf(x):
    loc = x.mean()
    scale = x.std()
    n = x.size
    pos = norm.cdf(x, loc, scale) * n
    return pos


class RmiModel:
    def __init__(self):
        self.level = [1, 4]  # 每层模型数

        # 数据相关信息
        self.data = None
        self.data_size = None
        self.mean = None  # 存储均值
        self.std = None  # 存储差标准差

        self.error_bound = []
        self.average_error = 0

        self.model = []  # 多层rmi的索引
        print("-------------begin building-----------------")
        self.build()
        print("---------------end building-----------------")

    # 初始化赋予模型，init里面自己调用
    def build(self):
        temp_model = SecondModel()
        self.model.append(temp_model)
        self.model.append([])
        for i in range(self.level[1]):
            temp_model = LinearRegression()
            self.model[1].append(temp_model)

    #fit和errbound的计算
    def train(self, data):
        self.data = data
        self.data_size = len(data)
        print('data here', data)
        y = cdf(data)
        print('y here', y)
        norm_data = preprocessing.scale(data)  # 标准化
        self.mean = data.mean()
        self.std = data.std()
        for m in self.level:
            if m == 1:
                # 先训练第一层 NN 16x16 Model
                self.model[0].fit(norm_data, y)
            else:
                sub_data = [[] for _ in range(m)]
                sub_y = [[] for _ in range(m)]
                for i in range(self.data_size):
                    # 按比例分配到四个下级机器学习模型之中
                    mm = int(self.model[0].predict([[norm_data[i]]]) * m / self.data_size)
                    if mm < 0:
                        mm = 0
                    elif mm > m - 1:
                        mm = m - 1
                    sub_data[mm].append(data[i])
                    sub_y[mm].append(y[i])
                # 训练第二层 SLR 模型
                for j in range(m):
                    xx = np.array(sub_data[j])
                    yy = np.array(sub_y[j])
                    min_err = max_err = 0
                    if xx.size > 0:
                        xx = np.reshape(xx, (-1, 1))
                        self.model[1][j].fit(xx, yy)
                        # 计算最后一层 Model 的 min_err/max_err
                        for i in range(data.size):
                            ppos, _ = self.predict(data[i])
                            err = ppos - i
                            self.average_error += abs(err)
                            if err < min_err:
                                min_err = math.floor(err)
                            elif err > max_err:
                                max_err = math.ceil(err)
                        self.average_error /= data.size
                        print(f"average error:{self.average_error / self.data_size * 100}%")
                    self.error_bound.append([min_err, max_err])

    def search(self, key):
        pos, model_index = self.predict(key)
        lp = pos + self.error_bound[model_index][0]
        rp = pos + self.error_bound[model_index][1]
        # 检查预测的位置是否超过范围
        if pos < 0:
            lp = pos = 0
        if pos > self.data_size - 1:
            rp = pos = self.data_size - 1
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

    def predict(self, key):
        mm = int(self.model[0].predict([[(key - self.mean) / self.std]]) * self.level[1] / self.data_size)
        if mm < 0:
            mm = 0
        elif mm > self.level[1] - 1:
            mm = self.level[1] - 1
        pos = int(self.model[1][mm].predict([[key]]))
        return pos, mm


def Test_Simple_main():
    data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    data = np.random.choice(data, 1000000)
    data.sort()
    print("@positive search:")
    data_test = np.random.choice(data, 10000)

    data_test = np.arange(100)
    data_test = data_test+285
    data_test = data_test.reshape(-1,1)
    data_test = torch.Tensor(data_test)

    rmi_test = RmiModel()
    rmi_test.train(data_test)



Test_Simple_main()

