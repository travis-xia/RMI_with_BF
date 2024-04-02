import pickle
import time
from rbtree import RBTree
from rbtree import RBNode
import numpy as np
with open('tree.pkl', 'rb') as f:
    loaded_tree = pickle.load(f)


data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
data = np.random.choice(data, 1000000)
count_suc = 0
count_fail = 0
print("begin search")
data_test = np.random.choice(data, 100000)
start = time.time()
for i in range(data_test.size):
    node_ans = loaded_tree.get_node(data_test[i])
    if node_ans==-1:
        count_fail += 1
        continue
    if node_ans.val == data_test[i] :
        pos = node_ans.pos
        count_suc +=1
    else:
        count_fail +=1
end = time.time()
print("search time:", end - start)
print("suc:",count_suc,"fail:",count_fail)


data = np.fromfile('1MBdata.txt', dtype=np.int32)
count_suc = 0
count_fail = 0
data_test = np.random.choice(data, 100000)
start = time.time()
for i in range(data_test.size):
    node_ans = loaded_tree.get_node(data_test[i])
    if node_ans==-1:
        count_fail += 1
        continue
    if node_ans.val == data_test[i] :
        pos = node_ans.pos
        count_suc +=1
    else:

        count_fail +=1
end = time.time()
print("search time:", end - start)
print("suc:",count_suc,"fail:",count_fail)

