# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import time as time
from Node import *
from memory_profiler import profile

def fun1(data,bf_level1):
    for i in data:
        bit = i>>30
        bf_level1[bit].add(i)
    return bf_level1
def fun2(dataset_for_neg_flush,bf_level1,clist,bf_level2):
    for i in dataset_for_neg_flush:
        bit = i >> 30
        flag = i >> 29
        if (i in bf_level1[bit]):
            clist[flag]+=1
            bf_level2[flag].add(i)
    return bf_level2,clist

@profile(precision=4)
def Test_Simple_main():
    MB =  int(input("请输入数据大小(1-200)MB:"))
    data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    data_size_ = MB *1000000
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
    print()


    start = time.time()
    fpr = 0.01
    la = BloomFilter(length[0]+length[1]+10000,fpr)
    lb = BloomFilter(length[2]+length[3]+10000,fpr)
    l0 = BloomFilter( int(data_size_*fpr)+10000,0.1)
    l1 = BloomFilter( int(data_size_*fpr)+10000,0.1)
    l2 = BloomFilter( int(data_size_*fpr)+10000,0.1)
    l3 = BloomFilter( int(data_size_*fpr)+10000,0.1)
    bf_level1 = []
    bf_level2 = []
    bf_level1.append(la)
    bf_level1.append(lb)
    bf_level2.append(l0)
    bf_level2.append(l1)
    bf_level2.append(l2)
    bf_level2.append(l3)
    bf_level1=fun1(data,bf_level1)
    dataset_for_neg_flush = np.random.randint(0,4<<29,data_size_)
    dataset_for_neg_flush = np.setdiff1d(dataset_for_neg_flush, data)#去重
    clist = [0,0,0,0]
    # for i in dataset_for_neg_flush:
    #     bit = i >> 30
    #     flag = i >> 29
    #     if (i in bf_level1[bit]):
    #         clist[flag]+=1
    #         bf_level2[flag].add(i)
    bf_level2,clist=fun2(dataset_for_neg_flush,bf_level1,clist,bf_level2)
    end = time.time()
    print(clist)
    print("Build pybloom model,time cost:", end - start)
    print()


    print("@positive search:")
    data_test = np.random.choice(data, 1)
    del data

    for i in data_test:
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
            pass
        else:
            ret = models[flag].search(i)



if __name__ == '__main__':
    Test_Simple_main();


