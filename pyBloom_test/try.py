# import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
#
# x_train = np.linspace(-20, 20, 10000)
# y_train = x_train * np.sin(x_train) + np.cos(x_train)
#
# model = keras.Sequential(
#     [
#         layers.Dense(10, activation="relu", input_shape=(1,), name="layer1"),
#         layers.Dense(10, activation="relu", name="layer2"),
#         layers.Dense(6, activation="relu", name="layer3"),
#         layers.Dense(1,activation="softmax", name="layer4"),
#     ]
# )
# model.compile(loss='mse', optimizer='sgd',
#               metrics=['mae'])
# model.summary()
# hist = model.fit(x_train,y_train,epochs=500)






import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
x_train = np.linspace(-5, 5, 1000,dtype=np.float32).reshape(-1,1)
y_train = x_train**2
x_train = torch.autograd.Variable(torch.from_numpy(x_train))
y_train = torch.autograd.Variable(torch.from_numpy(y_train))
print(np.hstack((x_train,y_train)))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label="original data")
plt.show()

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 12),
            nn.ReLU(),
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


def train(data,label,model,criterion,optimizer):

    #loss computing
    pred = model(data)
    loss = criterion(pred,label)

    #BP
    optimizer.zero_grad()
    loss.backward()# 梯度计算
    optimizer.step()# 梯度优化

    loss = loss.item()
    print(f"loss: {loss:>7f} ")
        # if i % 100 == 0:
        #     loss_rate = loss.item()
        #     print(f"loss: {loss_rate:>7f}  [{i:>5d}/{numOfData:>5d}]")


def test(data,label,model,criterion,flag):
    numOfData = len(data)
    correct= 0
    with torch.no_grad():
        pred = model(data)
        print(np.hstack((data,pred,label)))
        for i in range(numOfData):
            correct += 1 if (pred[i][0]==label[i][0]) else 0
        correct/=numOfData
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%\n")
    if flag:
        plt.cla()
        plt.plot(x_train.numpy(), pred.numpy(), label='Fitting Line')
        plt.show()

epochs = 5000
for i in range(epochs):
    flag = 0
    if i%500==0:
        flag = 1
    print(f"Epoch {i+1}\n-------------------------------")
    train(x_train,y_train, model, criterion, optimizer)
    test(x_train,y_train, model, criterion,flag)

print("Done!")


# def test_(dataloader, criterion):
#     model = LinearRegression()
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")








# num_epochs = 8000
# for epoch in range(num_epochs):
#     # if torch.cuda.is_available():
#     #   input = torch.autograd.Variable(x_train).cuda()
#     #   target = torch.autograd.Variable(y_train).cuda()
#     # else:
#     input = torch.autograd.Variable(x_train)
#     target = torch.autograd.Variable(y_train)
#     out = model(input)
#     loss = criterion(out, target)
#     optimizer.zero_grad()  # 清除上一梯度
#     loss.backward()  # 梯度计算
#
#     optimizer.step()  # 梯度优化
#     if (epoch + 1) % 200 == 0:
#         predict = model(torch.autograd.Variable(x_train))
#         predict = predict.data.numpy()
#         plt.cla()
#         plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label="original data")
#         plt.plot(x_train.numpy(), predict, label='Fitting Line')
#         plt.text(20, 0, 'Epoch[{}/{}],loss:{:.4f}'.format(epoch, num_epochs, loss.item()),
#                  fontdict={'size': 12, 'color': 'red'})
#         plt.pause(0.1)
# plt.ioff()
# plt.show()
# # print('Epoch[{}/{}],loss:{:.4f}'.format(epoch, num_epochs,loss.item()))
#


