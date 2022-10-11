import numpy as np
import pandas as pd
import time as time
from Newstructure import Filter
from Node import *
import sys
from pympler import tracker

# 人工生成的数据集测试
def Test_Simple_main():

    data_from = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    data = np.random.choice(data_from, 150000000)
    del data_from
    data.sort()

    print("~~data loaded,ready to separate")
    list0 = data[np.where(data < 1 << 30)[0]]
    list1 = data[np.where((data < 2 << 30) & (data >= 1 << 30))[0]]
    list2 = data[np.where((data < 3 << 30) & (data >= 2 << 30))[0]]
    list3 = data[np.where(data >= 3 << 30)[0]]
    print("~~list build,ready to feed models")
    if list0.size != 0:
        model0 = MLmodel(list0)
        print("~~model0 build", model0.model.coef_, model0.model.intercept_)
    if list1.size != 0:
        model1 = MLmodel(list1)
        print("~~model1 build", model1.model.coef_, model1.model.intercept_)
    if list2.size != 0:
        model2 = MLmodel(list2)
        print("~~model2 build", model2.model.coef_, model2.model.intercept_)
    if list3.size != 0:
        model3 = MLmodel(list3)
        print("~~model3 build", model3.model.coef_, model3.model.intercept_)
    print("~~models ready")
    print("######################the Mcost of build a empty BF")
    filter2 = Filter(len(data),FPR=0.001)
    print("######################the Mcost of inserting those nums into BF")
    filter2.Build(data)

    del list3
    del list2
    del list1
    del list0

    # print("@positive search:")
    # data_test = np.random.choice(data, 100)
    # del data
    # start = time.time()
    # count_suc = 0
    # count_fal_bf = 0
    # count_fal = 0
    # for i in data_test:
    #     # print(filter2.Contains(i))
    #     flag = filter2.Contains(i)
    #     if flag == -1:
    #         count_fal_bf += 1
    #     elif flag == 0:
    #         ret = model0.search(i)
    #         if ret:
    #             count_suc += 1
    #         else:
    #             count_fal += 1
    #     elif flag == 1:
    #         if model1.search(i):
    #             count_suc += 1
    #         else:
    #             count_fal += 1
    #     elif flag == 10:
    #         if model2.search(i):
    #             count_suc += 1
    #         else:
    #             count_fal += 1
    #     elif flag == 11:
    #         if model3.search(i):
    #             count_suc += 1
    #         else:
    #             count_fal += 1
    # end = time.time()
    # print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    # print("new cost is ", end - start)


if __name__ == '__main__':
    Test_Simple_main();


