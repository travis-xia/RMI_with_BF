import numpy as np
import pandas as pd
import time as time
from Bloomfilter.BloomFilter import BloomFilter
from pyBloom.Newstructure import Filter
from Node import *

#人工生成的数据集测试
def Test_Simple_main():
    '''
        正常从指定路径读入数据：(key,value)
        data=pd.read_csv(path)
        or
        data=pd.read_excel(path)
    '''
    '''Randomly generate data from (0,800000)'''
    data = np.hstack(np.random.randint( ((-1)<<31) ,1<<31, size=150000))
    data.sort()
    '''Generate three different model:Traditional bloom filter/pyBloom/RMI
        filter1=BloomFilter(capacity=?,error_rate=?);
        filter2=Filter(FPR=?);
        filter3=PMI(...)
    '''

    '''Test each model's timecost and capacity cost'''
    '''Build cost:'''

    list1 = np.array([])
    list2 = np.array([])
    list3 = np.array([])
    list0 = np.array([])
    for i in data:  # range(10):
        # if (i & 1) and ((i >> 16) & 1):
        #     list0 = np.append(list0, i)
        # if (i & 1) and not ((i >> 16) & 1):
        #     list1 = np.append(list1, i)
        # elif not (i & 1) and ((i >> 16) & 1):
        #     list2 = np.append(list2, i)
        # else:
        #     list3 = np.append(list3, i)
        if (i>>30)&3==0:
            list0 = np.append(list0, i)
        elif (i>>30)&3==1:
            list1 = np.append(list1, i)
        elif (i>>30)&3==2:
            list2 = np.append(list2, i)
        elif (i>>30)&3==3:
            list3 = np.append(list3, i)
    if list0.size != 0:
        print("0 build")
        model0 = MLmodel(list0)
    if list1.size != 0:
        print("1 build")
        model1 = MLmodel(list1)
    if list2.size != 0:
        print("2 build")
        model2 = MLmodel(list2)
    if list3.size != 0:
        print("3 build")
        model3 = MLmodel(list3)

    start=time.time()
    filter1 = BloomFilter(capacity=200000, error_rate=0.001)
    for i in range(len(data)):
        filter1.add(data[i])
    end=time.time()
    print("Tranditional model cost:",end-start)

    start=time.time()
    filter2=Filter(FPR=0.001)
    filter2.Build(data)
    end=time.time()
    print("Pybloom model cost:", end - start)

    '''
    start=time.time()
    filter3=RMI(...)
    filter3.Build(data)
    end=time.time()
    print("RMI model cost:", end - start)
    '''

    '''Test FPR and search time of each model'''
    traditionalerr=0
    newerr=0
    start=time.time()
    # for i in range(300000):
    #     print(i)
    #     if((i in data)==True and (i in filter1)==False):
    #         traditionalerr+=1
    #     elif((i in data)==False and (i in filter1)==True):
    #         traditionalerr+=1
    #     else:
    #         continue
    end=time.time()
    print("total error is")
    print(traditionalerr)
    print("traditional cost is ")
    print(end-start)
    start=time.time()
    count_suc = 0
    count_fal_bf = 0
    count_fal = 0
    for i in data:
        # print(filter2.Contains(i))
        flag = filter2.Contains(i)
        if flag == -1 :
            #print("False")
            count_fal_bf+=1
        elif flag ==0:
            if model0.search(i) :
                count_suc+=1
            else:
                count_fal+=1
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
        else:
            if model3.search(i):
                count_suc += 1
            else:
                count_fal += 1
    print("suc:",count_suc," bf fal:",count_fal_bf," fal:",count_fal)
    end=time.time()
    print("total error is")
    print(newerr)
    print("new cost is ")
    print(end - start)

if __name__=='__main__':
    Test_Simple_main();