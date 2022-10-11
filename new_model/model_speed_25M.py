# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import time as time
from Newstructure import Filter
from Node import *
import sys
from pympler import tracker
# import matplotlib.pyplot as plt
def Test_Simple_main():
    data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    data_size_ = 25000000
    data = np.random.choice(data, data_size_)
    data.sort()
    print("data loaded,ready to separate")
    list0 = data[np.where(data < 1 << 29)[0]]
    list1 = data[np.where((data < 2 << 29) & (data >= 1 << 29))[0]]
    list2 = data[np.where((data < 3 << 29) & (data >= 2 << 29))[0]]
    list3 = data[np.where(data >= 3 << 29)[0]]
    list0.sort()
    list1.sort()
    list2.sort()
    list3.sort()
    # plt.plot(list3,np.arange(len(list3)))
    # plt.plot(list3,(list3-list3[0])**0.5)
    # plt.xlabel("key")
    # plt.ylabel("position")
    # plt.show()
    # exit(0)
    length =  [len(list0),len(list1),len(list2),len(list3)]
    print('length:',length)
    # print(list3)
    #tr = tracker.SummaryTracker()
    print("list build,ready to feed models")
    models = []
    if list0.size != 0:
        model0 = MLmodel(list0)
        models.append(model0)
        print("model0 build", model0.model.coef_, model0.model.intercept_)
    if list1.size != 0:
        model1 = MLmodel(list1)
        models.append(model1)
        print("model1 build", model1.model.coef_, model1.model.intercept_)
    if list2.size != 0:
        model2 = MLmodel(list2)
        models.append(model2)
        print("model2 build", model2.model.coef_, model2.model.intercept_)
    if list3.size != 0:
        model3 = MLmodel(list3,1)
        models.append(model3)
        print("model3 build", model3.model.coef_, model3.model.intercept_)
    print("models ready")
    start = time.time()
    # filter2 = Filter(len(data),FPR=0.001)
    # filter2.Build(data)


    la = BloomFilter(length[0]+length[1]+10000,0.1)
    lb = BloomFilter(length[2]+length[3]+10000,0.1)
    l0 = BloomFilter(length[0]+10000,0.01)
    l1 = BloomFilter(length[1]+10000,0.01)
    l2 = BloomFilter(length[2]+10000,0.01)
    l3 = BloomFilter(length[3]+10000,0.01)
    bf_level1 = []
    bf_level2 = []
    bf_level1.append(la)
    bf_level1.append(lb)
    bf_level2.append(l0)
    bf_level2.append(l1)
    bf_level2.append(l2)
    bf_level2.append(l3)
    for i in data:
        bit = i>>30
        flag = i>>29
        bf_level1[bit].add(i)
        bf_level2[flag].add(i)
    end = time.time()
    print("Build pybloom model,time cost:", end - start)

    print("@positive search:")
    data_test = np.random.choice(data, 100000)
    data_test.sort()
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i not in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                count_fal += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    print("new cost is ", end - start)

    print("@25% negative search:")
    data_test = np.random.choice(data, 75000)
    data_test_ = np.random.randint(0,1<<20,size=25000)
    data_test.sort()
    data_test_.sort()
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i not in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                count_fal += 1
    for i in data_test_:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i not in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                count_fal += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    print("new cost is ", end - start)

    print("@50% negative search:")
    data_test = np.random.choice(data, 50000)
    data_test_ = np.random.randint(0,1<<20,size=50000)
    data_test.sort()
    data_test_.sort()
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i not in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                count_fal += 1
    for i in data_test_:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i not in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                count_fal += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    print("new cost is ", end - start)

    print("@75% negative search:")
    data_test = np.random.choice(data, 25000)
    data_test_ = np.random.randint(0, 1 << 20, size=75000)
    data_test.sort()
    data_test_.sort()
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i not in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                count_fal += 1
    for i in data_test_:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i not in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                count_fal += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    print("new cost is ", end - start)

    print("@100% negative search:")
    data_test_ = np.random.randint(0, 1 << 20, size=100000)
    data_test_.sort()
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test_:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i not in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                count_fal += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    print("new cost is ", end - start)

if __name__ == '__main__':
    Test_Simple_main();


