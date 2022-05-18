import struct
import  numpy as np
f=open('books_200M_uint32','rb')
data_list = list()

for i in range(200000000):
    if i%1000000==0:
        print(i)
    data = f.read(4)
    dataplus, = struct.unpack('I', data)
    # if i<20:
    #     print(dataplus)
    data_list.append(dataplus)

print(len(data_list))
f.close()
data_list = np.array(data_list)
data_list.tofile('books_200M_uint32.txt')