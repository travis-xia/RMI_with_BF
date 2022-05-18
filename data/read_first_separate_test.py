import numpy as np


data = np.array([])
data = np.fromfile('books_200M_uint32.txt', dtype=np.int32)
data.sort()
list0 = data[np.where(data < 1 << 30)[0]]
list1 = data[np.where((data < 2 << 30) & (data >= 1 << 30))[0]]
list2 = data[np.where((data < 3 << 30) & (data >= 2 << 30))[0]]
list3 = data[np.where(data >= 3 << 30)[0]]
print(len(list0),len(list1),len(list2),len(list3))


data = np.array([9,8,7,6,5,4,3,2,1,0])
print(data[np.where(data >5)[0]])
