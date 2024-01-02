import os
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# 优化器
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

if platform.system() == "Darwin":
    matplotlib.use("MacOSX")

lr = 0.001
xy = np.loadtxt(
    os.getcwd() + "/data/oil.csv", delimiter=",", skiprows=1, dtype=np.float32
)
x = xy[1:, :-3]  # 第一行为标签
y = xy[1:, -3:]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)


class dataset(Dataset):  # 需继承Dataset类并重写下面三个函数
    def __init__(self, xdata, ydata):
        self.xdata = torch.from_numpy(xdata)  # 创建张量
        self.ydata = torch.from_numpy(ydata)
        self.len = xdata.shape[0]

    def __getitem__(self, item):  # 索引函数
        return self.xdata[item], self.ydata[item]

    def __len__(self):
        return self.len


traindata = dataset(xtrain, ytrain)
testdata = dataset(xtest, ytest)
# 创建DataLoader，参数：数据集类、batchsize、是否打乱数据等
trainload = DataLoader(
    dataset=traindata, batch_size=32, shuffle=True, num_workers=0, drop_last=True
)
testload = DataLoader(
    dataset=testdata, batch_size=32, shuffle=False, num_workers=0, drop_last=True
)


class Model(torch.nn.Module):
    def __init__(
        self,
    ):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(13, 64)
        self.l2 = torch.nn.Linear(64, 32)
        self.l3 = torch.nn.Linear(32, 16)
        self.l4 = torch.nn.Linear(16, 8)
        self.l5 = torch.nn.Linear(8, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.l5(x)
        return x


model = Model()
# 激活器 https://github.com/jettify/pytorch-optimizer?tab=readme-ov-file
# 优化器 https://www.cnblogs.com/froml77/p/14956375.html
# criterion = torch.nn.BCELoss(reduction="sum")  # 损失函数
# optimizer = torch.optim.RAdam(model.parameters(), lr=0.04)  # 参数优化

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # SGD优化器

# criterion = torch.nn.BCELoss(reduction="mean")
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.06)

# criterion = torch.nn.CrossEntropyLoss()                             # 交叉熵
# optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=0.5)    # SGD优化器


def train(allepoch):  # 训练函数
    lepoch = []
    llsot = []
    lacc = []
    for epoch in range(allepoch):
        lost = 0
        l = 0
        for num, (x, y) in enumerate(trainload):
            y_h = model(x)
            loss = criterion(y_h, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lost += loss.item()
            l = num
        if epoch % 10 == 9:  # 每训练100次查看一下loss并测试查看准确率
            print("epoch:", epoch + 1, "loss:", lost / l, end=" ")
            acc = test()
            print("acc:", acc)
            lepoch.append(epoch / 10)
            llsot.append(lost / l)
            lacc.append(acc)
    plt.plot(lepoch, llsot, label="loss")
    plt.legend()
    plt.show()
    plt.plot(lepoch, lacc, label="acc")
    plt.show()


def test():  # 测试函数
    count = 0
    right = 0
    with torch.no_grad():  # 不参与参数优化
        for num, (x, y) in enumerate(testload):
            y_h = model(x)
            # print(y_h.size(), y.size(), y_h[0].size(), y[0].size())
            if y_h[0].size() != y[0].size():
                continue
            # logger.info(y_h.data)
            _, yhpred = torch.max(y_h.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度
            _, ypred = torch.max(y.data, dim=1)
            # if (yhpred != ypred).sum().item() > 0:
            #     logger.info(yhpred)
            #     logger.info(ypred)
            rightCount = (yhpred == ypred).sum().item()
            # logger.info(rightCount)
            right += rightCount
            count += y.size(0)
    return right / count  # 返回准确率


if __name__ == "__main__":
    train(2000)
