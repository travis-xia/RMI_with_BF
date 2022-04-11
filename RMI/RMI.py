import gc
import os
import time
import math
import keras
from memory_profiler import profile

#import BTREE.btree
import numpy as np
from scipy.stats import norm
from keras.layers import Dense
import scipy.interpolate as spi
from sklearn import preprocessing
from keras.models import Sequential
from sklearn.linear_model import LinearRegression

# import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class RMI:
    def __init__(self):
        self.RMI = [1, 4]
        self.index = []  # 存储模型的索引
        self.N = None  # key值总数量
        self.pos = None  # key对应的pos值
        self.data = None  # 所有key值
        self.ipo3 = None
        self.iy3 = None
        self.average_error = 0
        self.error_bound = []  # 存储最后一层每个Model的min_err and max_err
        self.mean = None  # 存储均值
        self.std = None  # 存储差标准差
        print("______________start Learned NN______________")
        start = time.time()
        self.build()
        end = time.time()
        print("Learned NN build time: ", end - start)
        print("============================================")

    def build(self):
        for m in self.RMI:
            if m == 1:
                # 第一层 NN Model 16x16
                model = Sequential()
                model.add(Dense(16, input_dim=1, activation="relu"))
                model.add(Dense(16, activation="relu"))
                model.add(Dense(1))
                #sgd = keras.optimizers.SGD(lr=0.000001)  # lr:学习率暂定
                model.compile(loss="mse", optimizer="sgd", metrics=["mse"])
                self.index.append(model)
                print(model)
                # 如果固定每个模型处理的块大小，需计算下一层模型的数量
                # x = np.arange(min(self.data), max(self.data), 0.1)
                # # 进行三次样条拟合
                # self.ipo3 = spi.splrep(self.pos, self.data, k=3)  # 样本点导入，生成参数
                # self.iy3 = spi.splev(x, self.ipo3)  # 根据观测点和样条参数，生成插值
                # self.index.append(self.iy3)
            # 第二层 => 建立多个线性回归模型
            else:
                self.index.append([])
                for j in range(m):
                    model = LinearRegression()
                    self.index[1].append(model)

    def train(self, data):
        self.data = data
        self.N = data.size
        y = cdf(data)
        norm_data = preprocessing.scale(data)  # 标准化
        self.mean = data.mean()
        self.std = data.std()
        for m in self.RMI:
            if m == 1:
                # 先训练第一层 NN 16x16 Model
                #sgd = keras.optimizers.SGD(lr=0.000001)
                self.index[0].compile(loss="mse", optimizer="sgd", metrics=["mse"])
                self.index[0].fit(norm_data, y, epochs=100, batch_size=32, verbose=0)
            else:
                # 根据训练结果分配到第二层
                sub_data = [[] for _ in range(m)]
                sub_y = [[] for _ in range(m)]
                for i in range(self.N):
                    mm = int(self.index[0].predict([[norm_data[i]]]) * m / self.N)
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
                    min_err = max_err = average_error = 0
                    if xx.size > 0:
                        xx = np.reshape(xx, (-1, 1))
                        self.index[1][j].fit(xx, yy)
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
                        print(f"average error:{self.average_error / self.N * 100}%")
                    self.error_bound.append([min_err, max_err])

    def predict(self, key):
        mm = int(self.index[0].predict([[(key - self.mean) / self.std]]) * self.RMI[1] / self.N)
        if mm < 0:
            mm = 0
        elif mm > self.RMI[1] - 1:
            mm = self.RMI[1] - 1
        ppos = int(self.index[1][mm].predict([[key]]))
        return ppos, mm

    @profile
    def search(self, key):  # model biased search
        start = time.time()
        pos, model = self.predict(key)
        lp = pos + self.error_bound[model][0]
        rp = pos + self.error_bound[model][1]
        # 检查预测的位置是否超过范围
        if pos < 0:
            lp = pos = 0
        if pos > self.N - 1:
            rp = pos = self.N - 1
        if lp < 0:
            lp = 0
        if rp > self.N - 1:
            rp = self.N - 1
        # print(l,pos,r)
        while lp <= rp:
            if self.data[pos] == key:
                end = time.time()
                print("learned NN search time: ", end - start)
                return pos
            elif self.data[pos] > key:
                rp = pos - 1
            elif self.data[pos] < key:
                lp = pos + 1
            pos = int((lp + rp) / 2)
        end = time.time()
        print("learned NN search time: ", end - start)
        return False


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


if __name__ == '__main__':
    data = np.hstack((np.random.randint(20000000, size=1000000), np.random.normal(100000, 10, size=1000000),
                      np.random.uniform(1, 10000000, size=1000000), np.random.poisson(100000, size=1000000),
                      np.random.exponential(1000000, size=1000000)))
    # data = np.random.randint(1, 1000, size=100)
    data = np.sort(data).reshape(-1)
    file = np.savetxt("data.csv", data, delimiter=",")
    li = RMI()
    start = time.time()
    li.train(data)
    end = time.time()
    print(f"Train Time: {end - start}s")
    li.search(data[10])
    # for k in data:
    #     print(li.search(k))
