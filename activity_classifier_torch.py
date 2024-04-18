import torch.nn as nn
import torch

import numpy as np
import pandas as pd

# sklearn 相关的包
from sklearn.utils import shuffle
# 导入标签编码器
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


# 自定义一个神经网络类
class TorchModel(nn.Module):
    def __init__(self, input_size: int):
        ''''
        inputsize: 输入样本的特征数量
        '''
        super(TorchModel,self).__init__() # 调用父类构造器初始化神经网络
        
        # 线性层的输出节点个数 == 分类的类别数(6)
        # input_size 就是每一个样本的特征数量 (假设是20)
        # 假设我们有一个输入矩阵 X(row=n, col = 20), 一个线性层的权重矩阵W(row = 20, col = 6)
        # 线性层的作用就是计算 X*W +b, 其中X*W(row = n, col = 6), b是常数（偏置值）
        self.linear = nn.Linear(input_size, 6) 
        
        
        # 线性层输出矩阵的每一行都是一个分类向量，形如 [1.2, 3.4, 5.5 ,0.2, 0.8, 0.9]
        # 这些数乍一看是无规则的，没什么意义， 因此我们要用softmax函数给他过滤一下
        # 就变成了 [0.2, 0.2, 0.1, 0.05, 0.05, 0.4], 这就是一个概率分布，0.4是最大的概率，因此这个样本被分到第6类 
        self.activation = nn.Softmax(dim=1) # dim表示应用softmax的维度， dim=1表示对第二维， 也就是列应用softmax
        self.loss = nn.CrossEntropyLoss() # 交叉熵损失函数
        
    
    def forward(self, x, y:torch.Tensor=None):
        '''
        x: 我们输入的样本矩阵
        y: 样本的真实标签， 如果y给出了，我们需要返回loss， 如果
            没给出，我们直接返回模型预测的y_pred
        '''
        
        # 得到线性层的输出
        x = self.linear(x)
        
        # 放入激活函数得到6个类的概率分布
        # softmax做完以后， 结果会被自动放入one-hot 函数，转为 0-1矩阵
        # 为什么要用one-hot, 因为单纯的概率分布向量看着不是很直观，并且loss不是太好计算
        # [0.2, 0.2, 0.1, 0.05, 0.05, 0.4]  ===one-hot==> [0, 0, 0, 0, 0, 1]  ----- 代表分到了第6类
        # [0, 0, 1, 0, 0, 0]  ----- 代表分到了第三类
        
        # y_pred.shape = (n, 6)
        y_pred:torch.Tensor= self.activation(x)
        
        
        if y is not None:
            # 计算预测值和真实值之间的损失并返回
            
            y=y.long() # 转为LongTensor, 这是强制要求
            # 注意！！！ 预测值必须在前， 真实值必须在后，否则报错
            return self.loss(y_pred, y)
        else:
            # 直接返回预测值
            return y_pred
    
    
    


def build_dataset()->pd.DataFrame:
    
    '''
    创建训练集和测试集
    '''
    train: pd.DataFrame = pd.read_csv('./train.csv')
    test: pd.DataFrame = pd.read_csv('./test.csv')
    
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
    
    # plt打印数据分布 --- 略
    
    # 将数据集中的特征列和标签列分离开来
    # train.drop: 删除这两列， axis=1: 删除的是列
    X_train = pd.DataFrame(train.drop(['Activity','subject'],axis=1))
    # values：获取DataFrame列的值
    # astype(object): 将列中的值转为object类型
    Y_train_label = train.Activity.values.astype(object)
    X_test = pd.DataFrame(test.drop(['Activity','subject'],axis=1))
    Y_test_label = test.Activity.values.astype(object)
    
    # 将标签列Activity中的string映射成整数，这样模型才可以学习一个函数来预测这个整数
    # 读取activity列中的所有字符串到一个列表 ... 手动赋予编码0-5
    # 创建一个标签编码器对象
    le = LabelEncoder()
    Y_train: np.ndarray = le.fit_transform(Y_train_label)
    Y_test: np.ndarray = le.fit_transform(Y_test_label)

    
    # print(Y_train.shape)
    # print(Y_train)
    # print(type(Y_train))
    
    # 将所有数据集全部转为Tensor
    X_train = torch.FloatTensor(X_train.to_numpy())
    Y_train = torch.FloatTensor(Y_train)
    X_test = torch.FloatTensor(X_test.to_numpy())
    Y_test = torch.FloatTensor(Y_test)
    
    # print(Y_train)
    
    return  X_train, Y_train, X_test, Y_test



def predict():
    pass


# 测试每轮(epoch)模型的准确率
def evaluate(model:TorchModel):
    model.eval()
    
    # 导入测试集
    _, _, test_x, test_y = build_dataset()
    
    test_sample_size=len(test_x)
    correct, wrong = 0, 0 # 预测正确的个数， 预测错误的个数
    
    with torch.no_grad():
        # 预测值是一个one-hot的0-1向量，1所在的位置代表真实类别
        test_y_pred: torch.Tensor= model(test_x)
        
        print('test_y_pred:\n {test_y_pred}')
        print('test_y:\n {test_y}')
        for y, y_pred in zip(test_y, test_y_pred):
            y_pred_label = torch.argmax(y_pred).item()
            
            if y_pred_label == y.item(): # y是一个tensor中的元素， 你可以暂时把tensor看成矩阵
                correct+=1
            else:
                wrong+=1
    print(f'分类的正确率为：{correct/(correct+wrong)}')
    return correct/(correct+wrong)


def softmax():
    '''
    自定义的softmax实现, 有空再说
    '''
    pass



# 执行训练任务
def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 561  # 输入向量维度
    learning_rate = 0.001  # 学习率  
    
    model = TorchModel(input_size=input_size)

    # 创建优化器，负责 反向传播，梯度更新，梯度归零
    # Adam 是一种自适应学习率的优化算法，它结合了 Momentum 和 RMSprop 的优点。
    # Momentum 能够在相关方向加速 SGD (随机梯度下降)，抑制振荡，加快（损失函数）收敛；
    # RMSprop 能够调整学习率，使其在参数的不同维度上具有不同的更新速度，这对于非均匀的数据非常有用。

    # parameters: 接收模型参数
    optim = torch.optim.Adam(params = model.parameters(), lr=learning_rate)
    
    # 损失日志
    log = []
    
    # 创建训练集
    train_x, train_y, test_x, test_y = build_dataset()
    
    for epoch in range(epoch_num):
        model.train()
        # 本轮训练中，记录所有批次（batch）的平均损失
        watch_loss= []
        
        # 所有的样本可以分成 train_sample//batch_size 个批次 （batch）
        for i in range(train_sample//batch_size):
            # 取出当前批次的训练数据
            x_batch = train_x[i*(batch_size): (i+1)*batch_size]
            y_batch = train_y[i*(batch_size): (i+1)*batch_size]

            # # 将 DataFrame 转换为 Tensor
            # x_batch:torch.Tensor = torch.tensor(x_batch.values)
            # y_batch:torch.Tensor = torch.tensor(y_batch.values)
            # 隐式调用forward成员函数， 计算损失
            
            print(f'y_batch:\n{y_batch}')
            print(len(y_batch))
            loss = model(x_batch, y_batch)
            
            loss.backward()  # 计算梯度
            optim.step() # 更新参数
            
            model.zero_grad() # 梯度归零， 每一个批次只能用该批次的损失函数来计算梯度
            
            watch_loss.append(loss.item())   
        
        print(f'epoch #{epoch+1}, average loss = {np.mean(watch_loss):.2f}')
        acc=evaluate(model)
        log.append([acc,np.mean(watch_loss)])
    
    # 保存模型到本地文件
    # torch.save(model.state_dict(), "classifier1.pt")
    
    # 画出性能曲线
    print(log)
    plt.plot(range(len(log)), [x[0] for x in log], label = 'accuracy')
    plt.plot(range(len(log)),[x[1] for x in log], label = 'loss')
    
    # 这里后面我会加其他的性能指标
    plt.legend()
    plt.show()
    return  
    


if __name__ == '__main__':
    build_dataset()
    main()
    
