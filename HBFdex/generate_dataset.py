import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
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


data = np.random.randint(0,10000000,1000000)
mit_data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)



y = np.random.randint(0,2147480285,200*1000000,dtype=np.int32)
data = np.unique(data)
# y = mit_data
x = np.arange(len(y))
y.sort()

print(len(y))
plt.plot(y,x,label='straight')
plt.xlabel("key")
plt.ylabel("position")
plt.show()

y.tofile('straight.txt')

# y1 = np.random.triangular(0,1 << 29,int(0.25*0.5*20000),dtype=np.int32)
# y2 = np.random.randint(1 << 29,2 << 29,int(0.25*0.5*20000),dtype=np.int32)
# y3 = np.random.randint(2 << 29,3 << 29,int(0.25*0.5*20000),dtype=np.int32)
# y4 = np.random.randint(3 << 29,4 << 29,int(1.25*0.5*20000),dtype=np.int32)
# y1 = np.random.choice(mit_data[0:50000000],int(0.25*0.5*20000))
# y2 = np.random.choice(mit_data[50000000:100000000],int(0.25*0.5*20000))
# y3 = np.random.choice(mit_data[100000000:150000000],int(0.25*0.5*20000))
# y4 = np.random.choice(mit_data[150000000:200000000],int(10.25*0.5*20000))
#
# y = np.concatenate([y1,y2,y3,y4])
# x = np.arange(len(y))
# y.sort()
#
# print(len(y))
# plt.plot(y,x,label="straight")
# plt.xlabel("key")
# plt.ylabel("position")
# plt.show()

# y.tofile('log.txt')









