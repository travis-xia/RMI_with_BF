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


def linear_scale(arr, new_min, new_max):
    """将numpy数组线性缩放到指定范围"""
    old_min = np.min(arr)
    old_max = np.max(arr)

    # 使用线性变换进行缩放
    scaled_arr = ((arr - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return scaled_arr

data = np.random.randint(0,10000000,1000000)
mit_data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)



# y = np.random.randint(0,2147480285,200*1000000,dtype=np.int32)
# data = np.unique(data)
# # y = mit_data
# x = np.arange(len(y))
# y.sort()
#
# print(len(y))
# plt.plot(y,x,label='straight')
# plt.xlabel("key")
# plt.ylabel("position")
# plt.show()
#
# y.tofile('straight.txt')



data_size = 200*1000000


data = np.exp(np.linspace(0, np.log(1<<31), data_size))
data =data.astype(int)
data.tofile('log.txt')
print(len(data))
# 绘制图表
plot_size = 1000000
data = np.random.choice(data,plot_size)
data.sort()
plt.plot(data, np.arange(plot_size))
plt.xlabel('Numbers (0 to 2^31)')
plt.ylabel('Index of the Number')
plt.title('Distribution of Numbers')
plt.savefig('log.png')
# plt.show()




# data_size = 200*1000000
# y1 = np.random.triangular(0,1 << 29,1<<29,int(0.25*0.5*data_size))
# y2 = np.random.randint(1 << 29,2 << 29,int(0.25*0.5*data_size),dtype=np.int32)
# y3 = np.random.randint(2 << 29,3 << 29,int(0.25*0.5*data_size),dtype=np.int32)
# y4 = np.random.randint(3 << 29,4 << 29,int(1.25*0.5*data_size),dtype=np.int32)
#
# y = np.concatenate([y1,y2,y3,y4])
# y = np.sqrt(y)
# y = linear_scale(y,0,1<<31 -1)
# y = y.astype(int)
# print(len(y))
# y.tofile('expo.txt')
#
# y = np.random.choice(y,1000000)
# x = np.arange(len(y))
# y.sort()
#
# plt.plot(y,x,label="straight")
# plt.xlabel("key")
# plt.ylabel("position")
# plt.title('Distribution of Numbers')
# plt.savefig('expo.png')











