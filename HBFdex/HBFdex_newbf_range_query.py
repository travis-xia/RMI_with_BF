# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import time as time
from Node import *
import pybloof

def Test_Simple_main(bf_rate_input = 10):
    MB =  int(input("请输入数据大小(1-200)MB:"))
    range_len = int(int(input('range_len(*1000):'))*1000)
    # MB = 1
    # range_len = 5000
    data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    data_size_ = int(MB *1000000)
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
    bf_rate = bf_rate_input
    la = pybloof.UIntBloomFilter(size=(length[0]+length[1]+10000)*bf_rate,hashes=5)
    lb = pybloof.UIntBloomFilter(size=(length[2]+length[3]+10000)*bf_rate,hashes=5)
    # l0 = BloomFilter( int((length[0]+length[1])*bf_rate)+10000,0.1)
    # l1 = BloomFilter( int((length[0]+length[1])*bf_rate)+10000,0.1)
    # l2 = BloomFilter( int((length[2]+length[3])*bf_rate)+10000,0.1)
    # l3 = BloomFilter( int((length[2]+length[3])*bf_rate)+10000,0.1)
    l0 = pybloof.UIntBloomFilter( size=int(data_size_*bf_rate)+10000,hashes=5)
    l1 = pybloof.UIntBloomFilter( size=int(data_size_*bf_rate)+10000,hashes=5)
    l2 = pybloof.UIntBloomFilter( size=int(data_size_*bf_rate)+10000,hashes=5)
    l3 = pybloof.UIntBloomFilter( size=int(data_size_*bf_rate)+10000,hashes=5)
    bf_level1 = []
    bf_level2 = []
    bf_level1.append(la)
    bf_level1.append(lb)
    bf_level2.append(l0)
    bf_level2.append(l1)
    bf_level2.append(l2)
    bf_level2.append(l3)
    lists = [list0,list1,list2,list3]
    for i in range(2):
        bf_level1[i].extend(lists[2*i])
        bf_level1[i].extend(lists[2*i+1])

    dataset_for_neg_flush = np.random.randint(0,4<<29,data_size_)
    dataset_for_neg_flush = np.setdiff1d(dataset_for_neg_flush, data)#去重

    neg_list0 = dataset_for_neg_flush[np.where(dataset_for_neg_flush < 1 << 29)[0]]
    neg_list1 = dataset_for_neg_flush[np.where((dataset_for_neg_flush < 2 << 29) & (dataset_for_neg_flush >= 1 << 29))[0]]
    neg_list2 = dataset_for_neg_flush[np.where((dataset_for_neg_flush < 3 << 29) & (dataset_for_neg_flush >= 2 << 29))[0]]
    neg_list3 = dataset_for_neg_flush[np.where(dataset_for_neg_flush >= 3 << 29)[0]]
    neg_lists = [neg_list0,neg_list1,neg_list2,neg_list3]
    print(len(neg_lists[0]))
    for i in range(4):
        bf_level2[i].extend(neg_lists[i])
    end = time.time()
    print("Build pybloom model,time cost:", end - start)
    print()


    print("range query test")
    data_test = np.random.randint(data[0],data[-1], 1000)
    data_test.sort()
    no_data_in_this_range=0
    start = time.time()
    for i in data_test:
        print('begin point',i)
        range_ret_list = []
        print("  begin range return list len",len(range_ret_list))
        right_bound = i +range_len
        cursor = i
        found = 0
        while found ==0 and cursor<=right_bound:
            bit = cursor >> 30
            flag = cursor >> 29
            if (cursor not in bf_level1[bit]) or (cursor in bf_level2[flag]):
                cursor+=1
            else:
                ret_pos = models[flag].search(cursor)
                if ret_pos==False:
                    cursor+=1
                else:
                    found = 1
        if cursor>right_bound:
            print("  !no data in this range")
            no_data_in_this_range +=1
            continue
        # print("  第一个key距离左边界：",cursor-i)
        # if cursor-i>range_len:
        #     print("--有问题,cursor,right_bound,range_len,i:",cursor,right_bound,range_len,i)
        range_ret_list.append(cursor)
        model_len = models[flag].data_size
        while found ==1 :
            ret_pos+=1
            if ret_pos>=model_len:
                break
            cursor=models[flag].data[ret_pos]
            if cursor>right_bound:
                break
            range_ret_list.append(cursor)
        print('  range ret list len: ',len(range_ret_list))
    end = time.time()
    # print('last range ret list len',len(range_ret_list))
    print("range cost is ", end - start)
    print('no_data_in_this_range:',no_data_in_this_range)



    # print("@positive search:")
    # data_test = np.random.choice(data, 100000)
    # data_test.sort()
    # start = time.time()
    # count_suc = 0
    # count_fal_bf = 0
    # count_fal = 0
    # for i in data_test:
    #     # print(filter2.Contains(i))
    #     bit = i >> 30
    #     flag = i >> 29
    #     if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
    #         # print("False")
    #         count_fal_bf += 1
    #     else:
    #         ret = models[flag].search(i)
    #         if ret:
    #             count_suc += 1
    #         else:
    #             count_fal += 1
    # end = time.time()
    # print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    # print("new cost is ", end - start)
    #
    # print("@25% negative search:")
    # data_test = np.random.choice(data, 75000)
    # data_test_ = np.random.randint(0, 1 << 20, size=25000)
    # data_test.sort()
    # data_test_.sort()
    # start = time.time()
    # count_suc = 0
    # count_fal_bf = 0
    # count_fal = 0
    # for i in data_test:
    #     # print(filter2.Contains(i))
    #     bit = i >> 30
    #     flag = i >> 29
    #     if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
    #         # print("False")
    #         count_fal_bf += 1
    #     else:
    #         ret = models[flag].search(i)
    #         if ret:
    #             count_suc += 1
    #         else:
    #             count_fal += 1
    # for i in data_test_:
    #     # print(filter2.Contains(i))
    #     bit = i >> 30
    #     flag = i >> 29
    #     if (i not in bf_level1[bit]) or (i in bf_level2[flag]):
    #         # print("False")
    #         count_fal_bf += 1
    #     else:
    #         ret = models[flag].search(i)
    #         if ret:
    #             count_suc += 1
    #         else:
    #             count_fal += 1
    # end = time.time()
    # print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)  # bf的fal也是准确的：我们准确地确认它不在
    # print("new cost is ", end - start)



if __name__ == '__main__':
    Test_Simple_main();


