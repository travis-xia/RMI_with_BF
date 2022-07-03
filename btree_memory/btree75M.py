import pandas as pd
import numpy as np
import csv
import random
import time
from pympler import tracker

from memory_profiler import profile


class BTreeNode:
    # degree:一个节点能容纳的字段数 initial
    def __init__(self, degree=2, number_of_keys=0, is_leaf=True, items=None, children=None, index=None):
        self.isLeaf = is_leaf
        self.numberOfKeys = number_of_keys
        # file number
        self.index = index
        if items is not None:
            self.items = items
        else:
            # 生成一个None列表,存放文件的编号
            self.items = [None] * (degree * 2 - 1)
        if children is not None:
            # 存放孩子文件的编号
            self.children = children
        else:
            self.children = [None] * degree * 2

    # set file number
    def set_index(self, index):
        self.index = index

    # get file number
    def get_index(self):
        return self.index

    # search in this node
    def search(self, b_tree, an_item):
        i = 0
        while i < self.numberOfKeys and an_item > self.items[i]:
            i += 1
        # 找到了目标节点
        if i < self.numberOfKeys and an_item == self.items[i]:
            return {'found': True, 'fileIndex': self.index, 'nodeIndex': i}
        if self.isLeaf:
            return {'found': False, 'fileIndex': self.index, 'nodeIndex': i - 1}
        else:
            # 根据孩子文件的编号在nodes里找到编号对应的节点
            return b_tree.get_node(self.children[i]).search(b_tree, an_item)


class BTree:
    # 初始化时只有根节点。 nodes为节点的map,映射filename:node freeindex:maybe是余量？
    def __init__(self, degree=2, nodes=None, root_index=1, free_index=2):
        if nodes is None:
            nodes = {}
        self.degree = degree
        if len(nodes) == 0:
            self.rootNode = BTreeNode(degree)
            self.nodes = {}
            self.rootNode.set_index(root_index)
            self.write_at(1, self.rootNode)
        else:
            self.nodes = nodes
            self.rootNode = self.nodes[root_index]
        self.rootIndex = root_index
        self.freeIndex = free_index

    # 得到下一个空闲filenum
    def get_free_index(self):
        self.freeIndex += 1
        return self.freeIndex - 1

    # new node?
    def get_free_node(self):
        new_node = BTreeNode(self.degree)
        index = self.get_free_index()
        new_node.set_index(index)
        self.write_at(index, new_node)
        return new_node

    # 在map中添加对应关系
    def write_at(self, index, a_node):
        self.nodes[index] = a_node

    # 分裂节点操作
    def split_child(self, p_node, i, c_node):
        new_node = self.get_free_node()
        new_node.isLeaf = c_node.isLeaf
        new_node.numberOfKeys = self.degree - 1
        for j in range(0, self.degree - 1):
            new_node.items[j] = c_node.items[j + self.degree]
        if c_node.isLeaf is False:
            for j in range(0, self.degree):
                new_node.children[j] = c_node.children[j + self.degree]
        c_node.numberOfKeys = self.degree - 1
        j = p_node.numberOfKeys + 1
        while j > i + 1:
            p_node.children[j + 1] = p_node.children[j]
            j -= 1
        p_node.children[j] = new_node.get_index()
        j = p_node.numberOfKeys
        while j > i:
            p_node.items[j + 1] = p_node.items[j]
            j -= 1
        p_node.items[i] = c_node.items[self.degree - 1]
        p_node.numberOfKeys += 1

    # build能写最前面的？？？
    def build(self, keys, values):
        if len(keys) != len(values):
            return
        for ind in range(len(keys)):
            self.insert(Item(keys[ind], values[ind]))

    def search(self, an_item):
        return self.rootNode.search(self, an_item)

    @profile(precision=4)
    def predict(self, key):
        start = time.time()
        search_result = self.search(Item(key, 0))
        a_node = self.nodes[search_result['fileIndex']]
        if a_node.items[search_result['nodeIndex']] is None:
            end = time.time()
            # print(f"predict time:{end - start}")
            return -1
        end = time.time()
        # print(f"predict time:{end - start}")
        return a_node.items[search_result['nodeIndex']].v

    def insert(self, an_item):
        search_result = self.search(an_item)
        if search_result['found']:
            return None
        r = self.rootNode
        if r.numberOfKeys == 2 * self.degree - 1:
            s = self.get_free_node()
            self.set_root_node(s)
            s.isLeaf = False
            s.numberOfKeys = 0
            s.children[0] = r.get_index()
            self.split_child(s, 0, r)
            self.insert_not_full(s, an_item)
        else:
            self.insert_not_full(r, an_item)

    def insert_not_full(self, inode, anitem):
        i = inode.numberOfKeys - 1
        if inode.isLeaf:
            while i >= 0 and anitem < inode.items[i]:
                inode.items[i + 1] = inode.items[i]
                i -= 1
            inode.items[i + 1] = anitem
            inode.numberOfKeys += 1
        else:
            while i >= 0 and anitem < inode.items[i]:
                i -= 1
            i += 1
            if self.get_node(inode.children[i]).numberOfKeys == 2 * self.degree - 1:
                self.split_child(inode, i, self.get_node(inode.children[i]))
                if anitem > inode.items[i]:
                    i += 1
            self.insert_not_full(self.get_node(inode.children[i]), anitem)

    def delete(self, an_item):
        an_item = Item(an_item, 0)
        search_result = self.search(an_item)
        if search_result['found'] is False:
            return None
        r = self.rootNode
        self.delete_in_node(r, an_item, search_result)

    # delete 后分类讨论
    def delete_in_node(self, a_node, an_item, search_result):
        if a_node.index == search_result['fileIndex']:
            i = search_result['nodeIndex']
            if a_node.isLeaf:
                while i < a_node.numberOfKeys - 1:
                    a_node.items[i] = a_node.items[i + 1]
                    i += 1
                a_node.numberOfKeys -= 1
            else:
                left = self.get_node(a_node.children[i])
                right = self.get_node(a_node.children[i + 1])
                if left.numberOfKeys >= self.degree:
                    a_node.items[i] = self.get_right_most(left)
                elif right.numberOfKeys >= self.degree:
                    a_node.items[i] = self.get_right_most(right)
                else:
                    k = left.numberOfKeys
                    left.items[left.numberOfKeys] = an_item
                    left.numberOfKeys += 1
                    for j in range(0, right.numberOfKeys):
                        left.items[left.numberOfKeys] = right.items[j]
                        left.numberOfKeys += 1
                    del self.nodes[right.get_index()]
                    for j in range(i, a_node.numberOfKeys - 1):
                        a_node.items[j] = a_node.items[j + 1]
                        a_node.children[j + 1] = a_node.children[j + 2]
                    a_node.numberOfKeys -= 1
                    if a_node.numberOfKeys == 0:
                        del self.nodes[a_node.get_index()]
                        self.set_root_node(left)
                    self.delete_in_node(left, an_item, {'found': True, 'fileIndex': left.index, 'nodeIndex': k})
        else:
            i = 0
            while i < a_node.numberOfKeys and self.get_node(a_node.children[i]).search(self, an_item)['found'] is False:
                i += 1
            c_node = self.get_node(a_node.children[i])
            if c_node.numberOfKeys < self.degree:
                j = i - 1
                while j < i + 2 and self.get_node(a_node.children[j]).numberOfKeys < self.degree:
                    j += 1
                if j == i - 1:
                    snode = self.get_node(a_node.children[j])
                    k = c_node.numberOfKeys
                    while k > 0:
                        c_node.items[k] = c_node.items[k - 1]
                        c_node.children[k + 1] = c_node.children[k]
                        k -= 1
                    c_node.children[1] = c_node.children[0]
                    c_node.items[0] = a_node.items[i - 1]
                    c_node.children[0] = snode.children[snode.numberOfKeys]
                    c_node.numberOfKeys += 1
                    a_node.items[i - 1] = snode.items[snode.numberOfKeys - 1]
                    snode.numberOfKeys -= 1
                elif j == i + 1:
                    snode = self.get_node(a_node.children[j])
                    c_node.items[c_node.numberOfKeys] = a_node.items[i]
                    c_node.children[c_node.numberOfKeys + 1] = snode.children[0]
                    a_node.items[i] = snode.items[0]
                    k = 0
                    for k in range(0, snode.numberOfKeys):
                        snode.items[k] = snode.items[k + 1]
                        snode.children[k] = snode.children[k + 1]
                    snode.children[k] = snode.children[k + 1]
                    snode.numberOfKeys -= 1
                else:
                    j = i + 1
                    snode = self.get_node(a_node.children[j])
                    c_node.items[c_node.numberOfKeys] = a_node.items[i]
                    c_node.numberOfKeys += 1
                    for k in range(0, snode.numberOfKeys):
                        c_node.items[c_node.numberOfKeys] = snode.items[k]
                        c_node.numberOfKeys += 1
                    del self.nodes[snode.index]
                    for k in range(i, a_node.numberOfKeys - 1):
                        a_node.items[i] = a_node.items[i + 1]
                        a_node.children[i + 1] = a_node.items[i + 2]
                    a_node.numberOfKeys -= 1
                    if a_node.numberOfKeys == 0:
                        del self.nodes[a_node.index]
                        self.set_root_node(c_node)
            self.delete_in_node(c_node, an_item, c_node.search(self, an_item))

    def get_right_most(self, anode):
        if anode.children[anode.numberOfKeys] is None:
            upitem = anode.items[anode.numberOfKeys - 1]
            self.delete_in_node(anode, upitem,
                                {'found': True, 'fileIndex': anode.index, 'nodeIndex': anode.numberOfKeys - 1})
            return upitem
        else:
            return self.get_right_most(self.get_node(anode.children[anode.numberOfKeys]))

    def set_root_node(self, r):
        self.rootNode = r
        self.rootIndex = self.rootNode.get_index()

    def get_node(self, index):
        return self.nodes[index]


# Value in Node
class Item:
    def __init__(self, k, v):
        self.k = k
        self.v = v

    def __gt__(self, other):
        if self.k > other.k:
            return True
        else:
            return False

    def __ge__(self, other):
        if self.k >= other.k:
            return True
        else:
            return False

    def __eq__(self, other):
        if self.k == other.k:
            return True
        else:
            return False

    def __le__(self, other):
        if self.k <= other.k:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.k < other.k:
            return True
        else:
            return False


# For Test
def b_tree_main():
    print("生成数据")
    tree_size = 75000000
    data_from = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
    data = np.random.choice(data_from, tree_size)
    del data_from
    data.sort()
    print("生成B树")
    b = BTree(2)
    print("插入数据")

    # tr = tracker.SummaryTracker()
    starttime = time.time()
    for i in range(data.size):
        if i % 100000 == 0:
            print(i)
        b.insert(Item(data[i], i))
    endtime = time.time()
    # tr.print_diff()
    print("building time is", endtime - starttime)


    data_test = np.random.choice(data, 1)
    del data
    for i in data_test:
        b.predict(i)
        break


if __name__ == '__main__':
    b_tree_main()
