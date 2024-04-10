# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import time as time
from Node import *
from btree import BTree, Item
from BTrees.OOBTree import OOBTree

def Test_Simple_main():
    MB =  int(input("请输入数据集数据大小(1-200)MB:"))
    MB_write =  int(input("请输入插入输入数据大小(1-200)MB:"))
    data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    data_size = MB *1000000
    data = np.random.choice(data, data_size)
    data_write = np.random.randint(0,3<<29,MB_write*1000000)
    whole_size = data_size + MB_write*1000000
    data.sort()
    data_write.sort()
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


    list0_write = data_write[np.where(data_write < 1 << 29)[0]]
    list1_write = data_write[np.where((data_write < 2 << 29) & (data_write >= 1 << 29))[0]]
    list2_write = data_write[np.where((data_write < 3 << 29) & (data_write >= 2 << 29))[0]]
    list3_write = data_write[np.where(data_write >= 3 << 29)[0]]

    start = time.time()
    fpr = 0.01
    la = BloomFilter(length[0]+length[1]+len(list0_write)+len(list1_write)+10000,fpr)
    lb = BloomFilter(length[2]+length[3]+len(list2_write)+len(list3_write)+10000,fpr)
    l0 = BloomFilter( int(whole_size*fpr)+10000,0.1)
    l1 = BloomFilter( int(whole_size*fpr)+10000,0.1)
    l2 = BloomFilter( int(whole_size*fpr)+10000,0.1)
    l3 = BloomFilter( int(whole_size*fpr)+10000,0.1)
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
        # bf_level2[flag].add(i)
    dataset_for_neg_flush = np.random.randint(0,4<<29,data_size)
    dataset_for_neg_flush = np.setdiff1d(dataset_for_neg_flush, data)#去重
    clist = [0,0,0,0]
    for i in dataset_for_neg_flush:
        bit = i >> 30
        flag = i >> 29
        if (i in bf_level1[bit]):
            clist[flag]+=1
            bf_level2[flag].add(i)
    end = time.time()
    print(clist)
    print("Build pybloom model,time cost:", end - start)
    print()



    btree1 =  OOBTree()
    btree2 = OOBTree()
    btree3 = OOBTree()
    btree4 =  OOBTree()
    blist = [btree1,btree2,btree3,btree4]
    write_list = [list0_write,list1_write,list2_write,list3_write]
    print("buffer loading:")
    for i in range(4):
        print('buffer',i)
        len_i = len(write_list[i])
        for ind in range(len_i):
            blist[i].update({write_list[i][ind]: ind})

    print('bloom filter inserting')
    for i in data_write:
        bit = i>>30
        flag = i>>29
        bf_level1[bit].add(i)
        # bf_level2[flag].add(i)


    print("@positive search:")
    data_test = np.random.choice(np.concatenate([data,data_write]), 100000)
    data_test.sort()
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                flag = blist[flag].has_key(i)
                if flag == False:
                    count_fal += 1
                else:
                    count_suc += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    print("new cost is ", end - start)

    print("@25% negative search:")
    data_test = np.random.choice(np.concatenate([data,data_write]), 75000)
    data_test_ = np.random.randint(0, 1 << 20, size=25000)
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
        if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                ret = models[flag].search(i)
                if ret:
                    count_suc += 1
                else:
                    flag = blist[flag].has_key(i)
                    if flag == False:
                        count_fal += 1
                    else:
                        count_suc += 1
    for i in data_test_:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                ret = models[flag].search(i)
                if ret:
                    count_suc += 1
                else:
                    flag = blist[flag].has_key(i)
                    if flag == False:
                        count_fal += 1
                    else:
                        count_suc += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    print("new cost is ", end - start)
    #
    print("@50% negative search:")
    data_test = np.random.choice(np.concatenate([data,data_write]), 50000)
    data_test_ = np.random.randint(0, 1 << 20, size=50000)
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
        if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                flag = blist[flag].has_key(i)
                if flag == False:
                    count_fal += 1
                else:
                    count_suc += 1
    for i in data_test_:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                flag = blist[flag].has_key(i)
                if flag == False:
                    count_fal += 1
                else:
                    count_suc += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    print("new cost is ", end - start)

    print("@75% negative search:")
    data_test = np.random.choice(np.concatenate([data,data_write]), 25000)
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
        if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                flag = blist[flag].has_key(i)
                if flag == False:
                    count_fal += 1
                else:
                    count_suc += 1
    for i in data_test_:
        # print(filter2.Contains(i))
        bit = i >> 30
        flag = i >> 29
        if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                flag = blist[flag].has_key(i)
                if flag == False:
                    count_fal += 1
                else:
                    count_suc += 1
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
        if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
            # print("False")
            count_fal_bf += 1
        else:
            ret = models[flag].search(i)
            if ret:
                count_suc += 1
            else:
                flag = blist[flag].has_key(i)
                if flag == False:
                    count_fal += 1
                else:
                    count_suc += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    print("new cost is ", end - start)


if __name__ == '__main__':
    Test_Simple_main();


