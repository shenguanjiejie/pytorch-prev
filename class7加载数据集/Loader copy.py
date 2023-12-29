# 数据加载
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset


# 准备数据集
class DiabetesDataset(Dataset):  # 抽象类DataSet
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        x = xy[1:, :-1]  # 第一行为标签
        y = xy[1:, [-1]]
        self.len = xy.shape[0]  # shape(多少行，多少列)
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)

        self.x_data = torch.from_numpy(xtrain)  # 创建张量
        self.y_data = torch.from_numpy(ytrain)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# dataset对象
dataset = DiabetesDataset(os.getcwd() + "/data/diabetes.csv")

# rangelist = list(range(0, 500))
# rangelist_test = list(range(500, len(dataset)))
dataset_test = Subset(dataset, range(500, len(dataset)))
last = dataset_test[len(dataset) - 500 - 1]
dataset = Subset(dataset, range(0, 500))
last = dataset[499]

# 使用DataLoader加载数据
train_loader = DataLoader(
    dataset=dataset,  # dataSet对象
    batch_size=32,  # 每个batch的条数
    shuffle=True,  # 是否打乱
    num_workers=4,
)  # 多线程一般设置为4和8

train_loader_test = DataLoader(
    dataset=dataset_test,  # dataSet对象
    batch_size=32,  # 每个batch的条数
    shuffle=True,  # 是否打乱
    num_workers=4,
)  # 多线程一般设置为4和8


# design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
        inputs, labels = data  # 取出一个batch
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        # 更新
        optimizer.step()

        # 每100个数据打印一次损失率
        running_loss += loss.item()
        if i % 100 == 0:
            logger.info("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    return running_loss


def test():  # 测试函数 https://blog.csdn.net/qq_43979221/article/details/130698816
    count = 0
    right = 0
    with torch.no_grad():  # 不参与参数优化
        for num, (x, y) in enumerate(train_loader_test):
            y_h = model(x)
            ypred = torch.where(y_h >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
            logger.info(ypred)
            right += (ypred == y).sum().item()
            count += y.size(0)
    return right / count  # 返回准确率


# def test():
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in train_loader_test:
#             xs, ys = data

#             for i in range(len(data)):
#                 logger.info(i)
#                 logger.info(xs[i])
#                 logger.info(ys[i])
#                 outputs = model(xs[i])
#                 # torch.max的返回值有两个，第一个是每一行的最大值是多少，第二个是每一行最大值的下标(索引)是多少。
#                 # _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度
#                 is1 = 0
#                 logger.info(outputs.data.numpy())
#                 if outputs.data > 0.7:
#                     is1 = 1
#                 correct += (is1 == ys[i]).sum().item()  # 张量之间的比较运算
#             total += ys.size(0)
#     logger.info("accuracy on test set: %d %% " % (100 * correct / total))


if __name__ == "__main__":
    lepoch = []
    llsot = []
    lacc = []
    for epoch in range(10):
        loss = train(epoch)
        acc = test()
        lepoch.append(epoch / 10)
        llsot.append(loss)
        lacc.append(acc)
    # plt.plot(lepoch, llsot, label="lost")
    plt.plot(lepoch, lacc, label="acc")
    plt.legend()
    plt.show()
