from pyBloom.Node import node

'''上层的filter结构'''
class Filter:
    '''初始化函数应该对层数以及期望的FPR率'''
    def __init__(self,FPR):
        '''我们的模型应该通过用户输入的层数和FPR推算出需要的bloom filter的大小和error_rate'''
        self.root=node(1,200000,0.01,200000,0.01)
        self.level = 2
        leftchd=node(2,200000,0.001,200000,0.001)
        rightchd = node(2, 200000, 0.001, 200000, 0.001)
        self.root.lefchld=leftchd
        self.root.rigchld=rightchd
    def Contains(self,num):
        temp=self.root
        result=0
        '''遍历这棵树'''
        for i in range(self.level):
            if temp.Contains(num):
                bias = num >> (32-i-1)
                flag = bias & 1
                if flag==1:
                    temp=temp.lefchld
                    result=result*10+1
                else:
                    temp=temp.rigchld
                    result=result*10
            else:
                # return False
                return -1

        ##此处改为合并学习model
        # return True
        return result
    def Add(self,num):
        temp = self.root
        # print("num is")
        # print(num)
        '''遍历这棵树'''
        for i in range(self.level):
            '''得到该层需要查找的数字'''
            # if (i == 0):
            #     print("第一层插入数字")
            # else:
            #     print("第二层插入数字")
            bias = num >> (32-i-1)
            flag = bias & 1
            # mask = ~((-1) << (16 + (i<<4)))
            # Toadd = num & mask
            # print(Toadd)
            '''根据1/0判断向哪边bloom filter添加'''
            if flag == 1:
                # print(temp.leftfilter==None)
                # judge=temp.Add(Toadd,1)
                judge = temp.Add(num, 1)
                '''判断是否重复'''
                # if judge:
                #     # print("元素已经存在！")
                #     temp=temp.lefchld
                # else:
                #     # print("元素第一次插入！")
                #     temp = temp.lefchld
                temp=temp.lefchld
                # print("向下移动")
            if flag == 0:
                # judge = temp.Add(Toadd,0)
                judge = temp.Add(num, 1)
                # if judge:
                #     # print("元素已经存在！")
                #     temp=temp.rigchld
                # else:
                #     # print("元素第一次插入！")
                #     temp=temp.rigchld
                temp=temp.rigchld
                # print("向下移动")
    def Build(self,keys):
        for ind in range(len(keys)):
            self.Add(keys[ind])
