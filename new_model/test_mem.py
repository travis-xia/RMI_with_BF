import numpy as np
import pandas as pd
import time as time
from Newstructure import Filter
from Node import *
import sys
from pympler import tracker

# 人工生成的数据集测试
def Test_Simple_main():

    '''
        正常从指定路径读入数据：(key,value)
        data=pd.read_csv(path)
        or
        data=pd.read_excel(path)
    '''
    '''Randomly generate data '''
    #data = np.hstack(np.random.randint(((-1) << 31), 1 << 31, size=50000))
    # data = np.arange(5000)
    data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    data = np.random.choice(data, 1000000)
    data.sort()
    '''Generate three different model:Traditional bloom filter/pyBloom/RMI
        filter1=BloomFilter(capacity=?,error_rate=?);
        filter2=Filter(FPR=?);
        filter3=RMI(...)
    '''

    '''Test each model's timecost and capacity cost'''
    '''Build cost:'''

    list1 = np.array([])
    list2 = np.array([])
    list3 = np.array([])
    list0 = np.array([])
    print("data loaded,ready to separate")

    list0 = data[np.where(data < 1 << 30)[0]]
    list1 = data[np.where((data < 2 << 30) & (data >= 1 << 30))[0]]
    list2 = data[np.where((data < 3 << 30) & (data >= 2 << 30))[0]]
    list3 = data[np.where(data >= 3 << 30)[0]]
    #tr = tracker.SummaryTracker()
    print("list build,ready to feed models")
    if list0.size != 0:
        model0 = MLmodel(list0)
        print("model0 build", model0.model.coef_, model0.model.intercept_)
    if list1.size != 0:
        model1 = MLmodel(list1)
        print("model1 build", model1.model.coef_, model1.model.intercept_)
    if list2.size != 0:
        model2 = MLmodel(list2)
        print("model2 build", model2.model.coef_, model2.model.intercept_)
    if list3.size != 0:
        model3 = MLmodel(list3)
        print("model3 build", model3.model.coef_, model3.model.intercept_)
    print("models ready")
    # compare two
    # start=time.time()
    # filter1 = BloomFilter(capacity=200000000, error_rate=0.001)
    # for i in range(len(data)):
    #     filter1.add(data[i])
    # end=time.time()
    # print("Building tranditional model cost:",end-start)
    start = time.time()
    filter2 = Filter(len(data),FPR=0.001)
    filter2.Build(data)
    end = time.time()
    print("Build pybloom model,time cost:", end - start)

    '''
    start=time.time()
    filter3=RMI(...)
    filter3.Build(data)
    end=time.time()
    print("RMI model cost:", end - start)
    '''

    '''Test FPR and search time of each model'''
    # 传统的暂不运行

    print("@positive search:")
    data_test = np.random.choice(data, 100000)
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test:
        # print(filter2.Contains(i))
        flag = filter2.Contains(i)
        if flag == -1:
            # print("False")
            count_fal_bf += 1
        elif flag == 0:
            ret = model0.search(i)
            if ret:
                count_suc += 1
            else:
                count_fal += 1
        elif flag == 1:
            if model1.search(i):
                count_suc += 1
            else:
                count_fal += 1
        elif flag == 10:
            if model2.search(i):
                count_suc += 1
            else:
                count_fal += 1
        elif flag==11:
            if model3.search(i):
                count_suc += 1
            else:
                count_fal += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)#bf的fal也是准确的：我们准确地确认它不在
    print("new cost is ",end - start)
    print("the correct rate:",(count_fal_bf+count_suc)*1.0/(count_fal_bf+count_suc+count_fal))

    print("@25% negative search:")
    data_test = np.hstack((np.random.randint(((-1) << 20),1<<20,size=25000), np.random.choice(data,75000)))
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test:
        # print(filter2.Contains(i))
        flag = filter2.Contains(i)
        if flag == -1:
            # print("False")
            count_fal_bf += 1
        elif flag == 0:
            ret = model0.search(i)
            if ret:
                # print("      found:",i,ret,list0[ret])
                count_suc += 1
            else:
                count_fal += 1
        elif flag == 1:
            if model1.search(i):
                count_suc += 1
            else:
                count_fal += 1
        elif flag == 10:
            if model2.search(i):
                count_suc += 1
            else:
                count_fal += 1
        elif flag==11:
            if model3.search(i):
                count_suc += 1
            else:
                count_fal += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)
    print("new cost is ",end - start)
    print("the correct rate:",(count_fal_bf+count_suc)*1.0/(count_fal_bf+count_suc+count_fal))

    print("@50% negative search:")
    data_test = np.hstack((np.random.randint(((-1) << 20),1<<20,size=50000), np.random.choice(data,50000)))
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test:
        # print(filter2.Contains(i))
        flag = filter2.Contains(i)
        if flag == -1:
            # print("False")
            count_fal_bf += 1
        elif flag == 0:
            ret = model0.search(i)
            if ret:
                # print("      found:",i,ret,list0[ret])
                count_suc += 1
            else:
                count_fal += 1
        elif flag == 1:
            if model1.search(i):
                count_suc += 1
            else:
                count_fal += 1
        elif flag == 10:
            if model2.search(i):
                count_suc += 1
            else:
                count_fal += 1
        elif flag==11:
            if model3.search(i):
                count_suc += 1
            else:
                count_fal += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)
    print("new cost is ",end - start)
    print("the correct rate:",(count_fal_bf+count_suc)*1.0/(count_fal_bf+count_suc+count_fal))

    print("@75% negative search:")
    data_test = np.hstack((np.random.randint(((-1) << 20),1<<20,size=75000), np.random.choice(data,25000)))
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test:
        # print(filter2.Contains(i))
        flag = filter2.Contains(i)
        if flag == -1:
            # print("False")
            count_fal_bf += 1
        elif flag == 0:
            ret = model0.search(i)
            if ret:
                # print("      found:",i,ret,list0[ret])
                count_suc += 1
            else:
                count_fal += 1
        elif flag == 1:
            if model1.search(i):
                count_suc += 1
            else:
                count_fal += 1
        elif flag == 10:
            if model2.search(i):
                count_suc += 1
            else:
                count_fal += 1
        elif flag==11:
            if model3.search(i):
                count_suc += 1
            else:
                count_fal += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)
    print("new cost is ",end - start)
    print("the correct rate:",(count_fal_bf+count_suc)*1.0/(count_fal_bf+count_suc+count_fal))

    print("@100% negative search:")
    data_test = np.hstack((np.random.randint(((-1) << 20),1<<20,size=100000)))
    start = time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data_test:
        # print(filter2.Contains(i))
        flag = filter2.Contains(i)
        if flag == -1:
            # print("False")
            count_fal_bf += 1
        elif flag == 0:
            ret = model0.search(i)
            if ret:
                # print("      found:",i,ret,list0[ret])
                count_suc += 1
            else:
                count_fal += 1
        elif flag == 1:
            if model1.search(i):
                count_suc += 1
            else:
                count_fal += 1
        elif flag == 10:
            if model2.search(i):
                count_suc += 1
            else:
                count_fal += 1
        elif flag==11:
            if model3.search(i):
                count_suc += 1
            else:
                count_fal += 1
    end = time.time()
    print("suc:", count_suc, " bf not in:", count_fal_bf, " fal:", count_fal)
    print("new cost is ",end - start)
    print("the correct rate:",(count_fal_bf+count_suc)*1.0/(count_fal_bf+count_suc+count_fal))

if __name__ == '__main__':
    Test_Simple_main();


