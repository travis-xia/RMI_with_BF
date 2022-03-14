from pyBloom.Node import node

'''上层的filter结构'''
class Filter:
    '''初始化函数应该对层数以及期望的FPR率'''
    def __init__(self,FPR):
        '''我们的模型应该通过用户输入的层数和FPR推算出需要的bloom filter的大小和error_rate'''
        self.root=node(1,1000,0.01,1000,0.01)
        self.level = 2
        leftchd=node(2,10000,0.001,10000,0.001)
        rightchd = node(2, 10000, 0.001, 10000, 0.001)
        self.root.lefchld=leftchd
        self.root.rigchld=rightchd
    def Contains(self,num):
        temp=self.root
        '''遍历这棵树'''
        for i in range(self.level):
            if temp.Contains(num):
                bias = num >> (i << 4)
                flag = bias & 1
                if flag==1:#是否>=2**(16i)
                    temp=temp.lefchld
                else:
                    temp=temp.rigchld
            else:
                return False
        return True
    def Add(self,num):
        temp = self.root
        '''遍历这棵树'''
        for i in range(self.level):
            '''得到该层需要查找的数字'''
            if (i == 0):
                print("第一层插入数字")
            else:
                print("第二层插入数字")
            bias = num >> (i << 4)
            flag = bias & 1
            mask = (1) << (16 + (i<<4))
            Toadd = num & mask
            #print(Toadd)
            '''根据1/0判断向哪边bloom filter添加'''
            if flag == 1:
                # print(temp.leftfilter==None)
                judge=temp.Add(Toadd,1)
                '''判断是否重复'''
                if judge:
                    print("元素已经存在！")
                    temp=temp.lefchld
                else:
                    print("元素第一次插入！")
                    temp = temp.lefchld
            if flag == 0:
                judge = temp.Add(Toadd,0)
                if judge:
                    print("元素已经存在！")
                    temp=temp.rigchld
                else:
                    print("元素第一次插入！")
                    temp=temp.rigchld







