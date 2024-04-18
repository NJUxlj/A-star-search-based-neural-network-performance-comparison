import torch.nn as nn
import torch

import numpy as np
import pandas as pd

# sklearn 相关的包
from sklearn.utils import shuffle

import matplotlib.pyplot as plt


# 自定义一个神经网络类
class TorchModel(nn.Module):
    def __init__(self, input_size: int):
        ''''
        inputsize: 输入样本的特征数量
        '''
        super(TorchModel,self).__init__() # 调用父类构造器
        
        # 线性层的输出节点个数 == 分类的类别数(6)
        # input_size 就是每一个样本的特征数量 (假设是20)
        # 假设我们有一个输入矩阵 X(row=n, col = 20), 一个线性层的权重矩阵W(row = 20, col = 6)
        # 线性层的作用就是计算 X*W +b, 其中X*W(row = n, col = 6), b是常数（偏置值）
        self.linear = nn.linear(input_size, 6)  
        self.activation = torch.softmax(dim=1) # dim表示应用softmax的维度， dim=1表示对第二维， 也就是列应用softmax
        self.loss = nn.CrossEntropyLoss() # 交叉熵损失函数
        
    
    def forward(self, x, y=None):
        '''
        x: 我们输入的样本矩阵
        y: 样本的真实标签， 如果y给出了，我们需要返回loss， 如果
            没给出，我们直接返回模型预测的y_pred
        '''
        
        # 得到线性层的输出
        x = self.linear(x)
        
        # 放入激活函数得到6个类的概率分布
        y_pred= self.activation(x)
        
        if y is not None:
            # 计算预测值和真实值之间的损失并返回
            return self.loss(y,y_pred)
        else:
            # 直接返回预测值
            return y_pred
    
    
    



def build_dataset():
    train: pd.DataFrame = shuffle(pd.read_csv('./train.csv'))
    test: pd.DataFrame = shuffle(pd.read_csv('./test.csv'))
    
    # 测试缺失值
    # isnull: 返回一个类型为bool的dataFrame
    # values: dataframe 转 numpy
    # any(): 有缺失值就返回True
    print("Does train has any missing value? %s"%train.isnull().values.any())
    
    # 填充缺失值
    if train.isnull().values.any():  
        # 使用每一列的平均值填充该列的缺失值， inplace: 原位修改
        train.fillna(train.mean(), inplace=True)
        test.fillna(test.mean(), inplace=True)
    
    
    print("===== 数据处理有点慢， 请耐心等待 =========")
    print(train.shape)
    
    # plt打印数据分布 --- 略
    
    # 将数据集中的特征列和标签列分离开来
    # train.drop: 删除这两列， axis=1: 删除的是列
    X_train = pd.DataFrame(train.drop(['Activity','subject'],axis=1))
    # values：获取DataFrame列的值
    # astype(object): 将列中的值转为object类型
    Y_train_label = train.Activity.values.astype(object)
    X_test = pd.DataFrame(test.drop(['Activity','subject'],axis=1))
    Y_test_label = test.Activity.values.astype(object)
    
    return X_train, Y_train_label, X_test, Y_test_label



def predict():
    pass


def evaluate():
    pass


def softmax():
    '''
    自定义的softmax实现
    '''
    pass



# 执行训练任务
def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 10  # 输入向量维度
    learning_rate = 0.001  # 学习率  
    
    model = TorchModel(input_size=input_size)

    # 创建优化器，负责 反向传播，梯度更新，梯度归零



if __name__ == '__main__':
    build_dataset()
