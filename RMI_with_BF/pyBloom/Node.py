'''node为整体树结构的节点，包含左右两个bloom filter,以及向子节点的指针'''
from Bloomfilter.BloomFilter import BloomFilter
class node:
    '''定义自己的左右两个bloom filter以及左右子节点'''
    def __init__(self,level,lefcap,leferr,rigcap,rigerr):
        self.level=level
        self.leftfilter=BloomFilter(lefcap,leferr)
        self.rightfilter=BloomFilter(rigcap,rigerr)
        self.lefchld=None
        self.rigchld=None
    '''提供查询数字是否存在的服务'''
    def Contains(self,num):
        bias=num>>((self.level-1)<<4)
        flag=bias & 1
        '''第一层取前16bits进入bloom filter,第二层取全部32bits'''
        mask = ~((-1)<<(16+((self.level-1)<<4)))
        temp= num & mask
        if flag==1:
            return temp in self.leftfilter
        else:
            return temp in self.rightfilter
    '''提供添加元素的服务'''
    def Add(self,num,flag):
        # print("开始添加数字"+str(num))
        if flag==1:
            return self.leftfilter.add(num)
        else:
            return self.rightfilter.add(num)
