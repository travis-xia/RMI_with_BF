# -*- coding: UTF-8 -*-
import math
import time

import numpy as np
from scipy.stats import norm
from model_lib import LinearRegression
from sklearn import preprocessing


class RmiModel:
    def __init__(self):
        self.level = [1, 4]  # 每层模型数

        # 数据相关信息
        self.data = None
        self.data_size = None
        self.lenOfAll = [0,0,0,0]
        self.mean = None  # 存储均值
        self.std = None  # 存储差标准差

        self.error_bound = []
        self.average_error = 0

        self.model = []  # 多层rmi的索引

        self.build()

    # 初始化赋予模型，init里面自己调用
    def build(self):
        temp_model = LinearRegression()
        self.model.append(temp_model)
        self.model.append([])
        for i in range(self.level[1]):
            temp_model = LinearRegression()
            self.model[1].append(temp_model)

    def train(self, data):
        self.data = data
        self.data_size = len(data)
        # print(self.data_size)
        y = cdf(data)
        # print('y here', y)
        # print(y[:100])
        # norm_data = preprocessing.scale(data)  # 标准化
        norm_data = (data - np.mean(data)) / np.std(data)
        # print(norm_data[:100])
        # print("len norm" , len(norm_data))
        self.mean = data.mean()
        self.std = data.std()
        for m in self.level:
            if m == 1:
                self.model[0].fit(norm_data, y)
            else:
                sub_data = [[] for _ in range(m)]
                sub_y = [[] for _ in range(m)]
                for i in range(self.data_size):
                    # 按比例分配到四个下级机器学习模型之中
                    mm = int(self.model[0].predict([[norm_data[i]]]) * m / self.data_size)
                    # print(i,mm,y[i])
                    if mm < 0:
                        mm = 0
                    elif mm > m - 1:
                        mm = m - 1
                    sub_data[mm].append(data[i])
                    sub_y[mm].append(y[i])
                # 训练第二层 模型
                for j in range(m):
                    print("    训练第二层model", j)
                    xx = np.array(sub_data[j])
                    yy = np.array(sub_y[j])
                    self.lenOfAll[j]=len(yy)
                    min_err = max_err = 0
                    if xx.size > 0:
                        xx = np.reshape(xx, (-1, 1))
                        self.model[1][j].fit(xx, yy)
                        # 计算最后一层 Model 的 min_err/max_err
                        print("calc err_b")
                        for i in range(data.size):
                            if j == m - 1 and i % 100000 == 0:
                                print("    ", i)
                            ppos, _ = self.predict(data[i])
                            err = i - ppos
                            self.average_error += abs(err)
                            if err < min_err:
                                min_err = math.floor(err)
                            elif err > max_err:
                                max_err = math.ceil(err)
                        self.average_error /= data.size
                        print("average error:", self.average_error* 1.0 / self.data_size)
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

    def predict(self, key):
        mm = int(self.model[0].predict([[(key - self.mean) / self.std]]) * self.level[1] / self.data_size)
        if mm < 0:
            mm = 0
        elif mm > self.level[1] - 1:
            mm = self.level[1] - 1
        pos = int(self.model[1][mm].predict([[key]]))
        return pos, mm


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


def Test_Simple_main():
    data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    # data = np.arange(1000000)+123
    data.sort()
    data_choice = np.random.choice(data, 1000000)
    data_choice.sort()
    # exit(0)
    # data_choice = data.copy()
    rmi_test = RmiModel()
    rmi_test.train(data_choice)

    print("@positive search:")
    data_test = np.random.choice(data_choice, 100000)
    # data_test = data_test.reshape(-1, 1)
    start_time = time.time()
    count_suc = 0
    count_fal = 0
    for i in data_test:
        if rmi_test.search(i):
            count_suc += 1
        else:
            count_fal += 1
    end_time = time.time()
    print("suc:", count_suc, " fal:", count_fal)
    print("the correct rate:", (count_suc) * 1.0 / (count_suc + count_fal))
    print("time cost:", end_time - start_time)

    print("@25% negative search:")
    data_test = np.random.choice(data_choice, 75000)
    data_test.sort()
    # data_test_ = np.random.choice(data_choice, 25000)
    data_test_ = data_test[:25000] +1234
    data_test_.sort()
    count_suc = 0
    count_fal = 0
    start_time = time.time()
    for i in data_test:
        if rmi_test.search(i):
            count_suc += 1
        else:
            count_fal += 1

    for i in data_test_:
        if rmi_test.search(i):
            count_suc += 1
        else:
            count_fal += 1
    end_time = time.time()
    print("suc:", count_suc, " fal:", count_fal)
    print("the correct rate:", (count_suc) * 1.0 / (count_suc + count_fal))
    print("time cost:", end_time - start_time)

    print("@50% negative search:")
    data_test = np.random.choice(data_choice, 50000)
    # data_test_ = np.random.randint(0, 1 << 20, size=50000)
    data_test_ = data_test+1234
    start_time = time.time()
    count_suc = 0
    count_fal = 0
    for i in data_test:
        if rmi_test.search(i):
            count_suc += 1
        else:
            count_fal += 1
    for i in data_test_:
        if rmi_test.search(i):
            count_suc += 1
        else:
            count_fal += 1
    end_time = time.time()
    print("suc:", count_suc, " fal:", count_fal)
    print("the correct rate:", (count_suc) * 1.0 / (count_suc + count_fal))
    print("time cost:", end_time - start_time)

    print("@75% negative search:")
    data_test = np.random.choice(data_choice, 25000)
    data_test_ = np.random.choice(data_choice, 75000) +1234
    start_time = time.time()
    count_suc = 0
    count_fal = 0
    for i in data_test:
        if rmi_test.search(i):
            count_suc += 1
        else:
            count_fal += 1
    for i in data_test_:
        if rmi_test.search(i):
            count_suc += 1
        else:
            count_fal += 1
    end_time = time.time()
    print("suc:", count_suc, " fal:", count_fal)
    print("the correct rate:", (count_suc) * 1.0 / (count_suc + count_fal))
    print("time cost:", end_time - start_time)

    print("@100% negative search:")
    data_test_ = np.random.choice(data_choice, 100000)+1234
    start_time = time.time()
    count_suc = 0
    count_fal = 0
    for i in data_test_:
        if rmi_test.search(i):
            count_suc += 1
        else:
            count_fal += 1
    end_time = time.time()
    print("suc:", count_suc, " fal:", count_fal)
    print("the correct rate:", (count_suc) * 1.0 / (count_suc + count_fal))
    print("time cost:", end_time - start_time)


if __name__ == '__main__':
    Test_Simple_main()
