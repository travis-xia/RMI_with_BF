import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

x_train_list = []
y_train_list = []
for i in range(1, 50):
    x = i * random.choice([0.7, 0.8, 0.9])
    y = i * random.choice([0.4, 0.5, 0.8, 0.9])
    x_train_list.append(["%.2f" % x])
    y_train_list.append(["%.2f" % y])

x_train = np.array(x_train_list, dtype=np.float32)  # 将数据列表转为np.array
y_train = np.array(y_train_list, dtype=np.float32)
print(x_train,y_train)
plt.scatter(x_train, y_train)
plt.show()
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 10)
        self.linear1 = nn.Linear(10, 10)
        self.linear1_1 = nn.Linear(10, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.linear(x)
        for _ in range(3):
            x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(x)
        x = nn.functional.relu(self.linear1_1(x))
        x = self.linear2(x)
        return x


# if torch.cuda.is_available():
# model = LinearRegression().cuda()
# else:
model = LinearRegression()
criterion = nn.MSELoss()
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=1e-3)
num_epochs = 8000
for epoch in range(num_epochs):
    # if torch.cuda.is_available():
    #   input = torch.autograd.Variable(x_train).cuda()
    #   target = torch.autograd.Variable(y_train).cuda()
    # else:
    input = torch.autograd.Variable(x_train)
    target = torch.autograd.Variable(y_train)
    out = model(input)
    loss = criterion(out, target)
    optimizer.zero_grad()  # 清除上一梯度
    loss.backward()  # 梯度计算

    optimizer.step()  # 梯度优化
    if (epoch + 1) % 200 == 0:
        predict = model(torch.autograd.Variable(x_train))
        predict = predict.data.numpy()
        plt.cla()
        plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label="original data")
        plt.plot(x_train.numpy(), predict, label='Fitting Line')
        plt.text(20, 0, 'Epoch[{}/{}],loss:{:.4f}'.format(epoch, num_epochs, loss.item()),
                 fontdict={'size': 12, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
# print('Epoch[{}/{}],loss:{:.4f}'.format(epoch, num_epochs,loss.item()))