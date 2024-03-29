from Node import node
from memory_profiler import profile

'''上层的filter结构'''
class Filter:
    '''初始化函数应该对层数以及期望的FPR率'''
    @profile
    def __init__(self,dataSize,FPR):
        '''我们的模型应该通过用户输入的层数和FPR推算出需要的bloom filter的大小和error_rate'''
        self.root=node(1,dataSize,0.001,dataSize,0.001)
        print("  root build")
        self.level = 2
        leftchd=node(2,dataSize,0.001,dataSize,0.001)
        print("  leftchd build")
        rightchd = node(2, dataSize, 0.001, dataSize, 0.001)
        print("  rightchd build")
        self.root.lefchld=leftchd
        self.root.rigchld=rightchd

    def Contains(self,num):
        temp=self.root
        result=0
        '''遍历这棵树'''

        for i in range(self.level):
            if temp.Contains(num):
                flag =( num >> (32-i-1)) & 1
                if flag==1:
                    temp=temp.lefchld
                    result=result*10+1
                else:
                    temp=temp.rigchld
                    result=result*10
            else:
                return -1

        return result
    def Add(self,num):
        temp = self.root
        '''遍历这棵树'''
        for i in range(self.level):
            '''得到该层需要查找的数字'''
            bias = num >> (32-i-1)
            flag = bias & 1
            '''根据1/0判断向哪边bloom filter添加'''
            if flag == 1:
                # print(temp.leftfilter==None)
                # judge=temp.Add(Toadd,1)
                judge = temp.Add(num, 1)
                '''判断是否重复'''
                temp=temp.lefchld
                # print("向下移动")
            if flag == 0:
                judge = temp.Add(num, 1)
                temp=temp.rigchld
                # print("向下移动")
    #@profile
    def Build(self,keys):
        length = len(keys)
        print("    BF inserting nums:")
        for ind in range(length):
            if ind %100000 ==0:
                print("     ",ind)
            self.Add(keys[ind])
