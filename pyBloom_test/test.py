import numpy as np

from pyBloom.Newstructure import Filter
from Node import *

Tree=Filter(0.001)
list1 = np.array([])
list2 = np.array([])
list3 = np.array([])
list0 = np.array([])


#def add_to_ml(x):

data = [1,2,3,4,5,17,19,1<<7,1<<13,1<<23,1<<33,1<<18]


for i in data:#range(10):
    Tree.Add(i)
    if (i & 1) and ((i>>16)&1):
        list0 = np.append(list0, i)
    if (i & 1) and not ((i>>16)&1):
        list1 = np.append(list1, i)
    elif not (i & 1) and ((i>>16)&1):
        list2 = np.append(list2, i)
    else:
        list3 = np.append(list3, i)

if list0.size!=0:
    model0 = MLmodel(list0)
if list1.size!=0:
    model1 = MLmodel(list1)
if list2.size!=0:
    model2 = MLmodel(list2)
if list3.size!=0:
    model3 = MLmodel(list3)


for i in range(20):
    #print(Tree.Contains(i))
    print(model3.search(i))
