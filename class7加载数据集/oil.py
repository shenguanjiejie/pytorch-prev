import os
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

if platform.system() == "Darwin":
    matplotlib.use("MacOSX")

lr = 0.001
xy = np.loadtxt(
    os.getcwd() + "/data/diabetes.csv", delimiter=",", skiprows=0, dtype=np.float32
)
x = xy[1:, :-1]  # 第一行为标签
y = xy[1:, [-1]]
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
        self.l1 = torch.nn.Linear(8, 32)
        self.l2 = torch.nn.Linear(32, 16)
        self.l3 = torch.nn.Linear(16, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.sig(x)
        x = self.l2(x)
        x = self.sig(x)
        x = self.l3(x)
        x = self.sig(x)
        return x


model = Model()
# criterion = torch.nn.BCELoss(reduction="sum")  # 损失函数
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 参数优化
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


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
    # plt.plot(lepoch, llsot, label="lost")
    plt.plot(lepoch, lacc, label="acc")
    plt.legend()
    plt.show()


def test():  # 测试函数
    count = 0
    right = 0
    with torch.no_grad():  # 不参与参数优化
        for num, (x, y) in enumerate(testload):
            y_h = model(x)
            # logger.info(y_h.data)
            ypred = torch.where(y_h >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
            logger.info(ypred)
            right += (ypred == y).sum().item()
            count += y.size(0)
    return right / count  # 返回准确率


if __name__ == "__main__":
    train(100)
