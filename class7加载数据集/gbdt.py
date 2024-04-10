import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import pandas as pd
import os
#分训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn.metrics import SCORERS

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False

CRCtrain=pd.read_excel(os.getcwd() + "/data/IBD血液原始数据_副本.xlsx",sheet_name="处理数据")#导入数据

# CRCtrain=CRCtrain.drop(["姓名","疾病"],axis=1) #将第一列Unnamed: 0删除

#codes 方法返回每个分类的整数编码。
#这些编码是从 0 开始分配的，表示每个不同的分类。
#例如，如果 '对照' 列包含三个不同的分类：'A', 'B', 'C'，则 'A' 的编码为 0，'B' 的编码为 1，'C' 的编码为 2。

CRCtrain['ill'] = pd.Categorical(CRCtrain['ill']).codes 
#将 '对照' 列转换为整数编码，并将结果存储回 '对照' 列。CRC2['类别'] = pd.Categorical(IBD_2['类别']).codes

#提取特征列
CRCtrainX=CRCtrain.iloc[:,0:54] ##定义x
#提取标签列 
CRCtrainY=CRCtrain.iloc[:,54]##定义标签

#模型训练
clfXGB=XGBClassifier(random_state=200)

X_train,X_test,y_train,y_test=train_test_split(CRCtrainX,CRCtrainY,test_size=0.2,random_state=100)

#在使用 ShuffleSplit 进行交叉验证时，通常需要指定 test_size 或 train_size，其中至少一个必须是非空的。

#例如，如果指定 test_size=0.3，则将数据集划分为 70% 的训练集和 30% 的测试集。

#每次迭代时，ShuffleSplit 都会重新随机重排数据集，并将其划分为训练集和测试集。

#最后，可以通过计算多个迭代的平均性能，来评估模型的泛化能力和稳定性。

# StratifiedShuffleSplit 是一种交叉验证策略，用于将数据集划分为训练集和测试集。与 ShuffleSplit 类似，StratifiedShuffleSplit 也是通过对数据集进行随机重排，然后将数据集划分为多个训练集和测试集，以测试模型在不同的训练集和测试集上的性能。不同之处在于 StratifiedShuffleSplit 会保证每个划分中，训练集和测试集中的类别分布是一致的，从而避免了因类别分布不均衡而导致的模型偏差或方差问题。


# sklearn.metrics.SCORERS.keys()返回一个包含所有可用的scoring函数名称的列表。这些名称可以作为scoring参数传递给cross_val_score()和GridSearchCV()等函数
# 'accuracy'：准确率；
# 'balanced_accuracy'：平衡准确率；
# 'top_k_accuracy'：top_k准确率，k可以通过参数进行设置；
# 'average_precision'：平均准确率；
# 'neg_brier_score'：负Brier分数；
# 'f1'：F1分数，综合考虑了精确率和召回率；
# 'f1_micro'：微平均F1分数；
# 'f1_macro'：宏平均F1分数；
# 'f1_weighted'：加权平均F1分数，按照样本数量加权；
# 'f1_samples'：样本平均F1分数，针对多标签分类；
# 'neg_log_loss'：负对数似然损失，适用于概率模型；
# 'precision'：精确率；
# 'recall'：召回率；
# 'roc_auc'：ROC曲线下的面积；
# 'roc_auc_ovo'：多类别ROC曲线下的面积（一对多）；
# 'roc_auc_ovr'：多类别ROC曲线下的面积（一对剩余）；
# 'balanced_accuracy'：平衡准确率。
# 需要注意的是，有些指标是多类别分类特有的，例如roc_auc_ovo和roc_auc_ovr，可以处理多类别分类的问题。

print(SCORERS.keys())
clfXGB.fit(CRCtrainX,CRCtrainY)
# 预测概率
CRCtrainY_prob = clfXGB.predict_proba(CRCtrainX)[:, 1]


