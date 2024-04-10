import os
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# 优化器
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

if platform.system() == "Darwin":
    matplotlib.use("MacOSX")

lr = 0.001
CRCtrain=pd.read_excel(os.getcwd() + "/data/IBD血液原始数据_副本.xlsx",sheet_name="处理数据")#导入数据

xy = CRCtrain.to_numpy()
x = xy[1:, :-1]  # 第一行为标签
y = xy[1:, -1:]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0,shuffle=True)


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
    dataset=traindata, batch_size=16, shuffle=True, num_workers=0, drop_last=True
)
testload = DataLoader(
    dataset=testdata, batch_size=16, shuffle=True, num_workers=0, drop_last=True
)


class Model(torch.nn.Module):
    def __init__(
        self,
    ):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(54, 512)
        self.l2 = torch.nn.Linear(512, 16)
        self.l3 = torch.nn.Linear(16, 1)
        # self.l4 = torch.nn.Linear(8, 1)
        # self.l5 = torch.nn.Linear(4, 1)
        self.ls = [self.l1,self.l2,self.l3]
        # self.ls = [self.l1,self.l2,self.l3,self.l4]
        # self.ls = [self.l1,self.l2,self.l3,self.l4,self.l5]
        self.relu = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        for l in self.ls:
            x = l(x)
            x = self.relu(x)
        return x


model = Model()
# 激活器 https://github.com/jettify/pytorch-optimizer?tab=readme-ov-file
# 优化器 https://www.cnblogs.com/froml77/p/14956375.html
# criterion = torch.nn.BCELoss(reduction="sum")  # 损失函数
# optimizer = torch.optim.RAdam(model.parameters(), lr=0.04)  # 参数优化

# criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
criterion =  torch.nn.BCELoss(reduction="mean")  #  会收敛 不多, 准确率 70%左右
# criterion =  torch.nn.L1Loss()  #
# optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.5)  # SGD优化器 85%
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.2)  # SGD优化器 85%

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
            y = y.float()
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
            
            for i in range(y_h.data.shape[0]):
                yv = y.data[i,0].item()
                if yv == (y_h.data[i,0].item() > 0.5)*1:
                    right +=1
                    
            # _, yhpred = torch.max(y_h.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度
            # _, ypred = torch.max(y.data, dim=1)
            # # if (yhpred != ypred).sum().item() > 0:
            # #     logger.info(yhpred)
            # #     logger.info(ypred)
            # rightCount = (yhpred == ypred).sum().item()
            # # logger.info(rightCount)
            # right += rightCount
            count += y.size(0)
    if count == 0:
        return 0
    return right / count  # 返回准确率


if __name__ == "__main__":
    train(10000)
