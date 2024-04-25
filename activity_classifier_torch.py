import torch.nn as nn
import torch

import numpy as np
import pandas as pd
from typing import Union
import typing


# sklearn related packages
from sklearn.utils import shuffle
# 导入标签编码器
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
# 导入混淆矩阵
from sklearn.metrics import confusion_matrix,classification_report, f1_score

from sklearn.metrics import roc_curve, auc, roc_auc_score,precision_score,recall_score

import matplotlib.pyplot as plt


from activity_classifier_sklearn import *

# self-defined neural network class
class TorchModel(nn.Module):
    def __init__(self, input_size: int=561, layers:int=4, units:int=15, hidden_activation:nn.Module=nn.Tanh):
        ''''
        inputsize: 输入样本的特征数量
        '''
        super(TorchModel,self).__init__() # 调用父类构造器初始化神经网络
        
        # 线性层的输出节点个数 == 分类的类别数(6)
        # input_size 就是每一个样本的特征数量 (假设是20)
        # 假设我们有一个输入矩阵 X(row=n, col = 20), 一个线性层的权重矩阵W(row = 20, col = 6)
        # 线性层的作用就是计算 X*W +b, 其中X*W(row = n, col = 6), b是常数（偏置值）
        self.linear_list=nn.ModuleList()
        
        self.hidden_activation = hidden_activation
        if layers<1:
            linear1 = nn.Linear(input_size, units)
            self.linear_list.append1(linear1)
        else:
            linear1 = nn.Linear(input_size, units) 
            self.linear_list.append(linear1)
            
            for i in range(1,layers-1):
                linear2 = nn.Linear(units, units)
                self.linear_list.append(linear2)
                
            linear3 = nn.Linear(units, 6) 
            self.linear_list.append(linear3)
        
        # 线性层输出矩阵的每一行都是一个分类向量，形如 [1.2, 3.4, 5.5 ,0.2, 0.8, 0.9]
        # 这些数乍一看是无规则的，没什么意义， 因此我们要用softmax函数给他过滤一下
        # 就变成了 [0.2, 0.2, 0.1, 0.05, 0.05, 0.4], 这就是一个概率分布，0.4是最大的概率，因此这个样本被分到第6类 
        self.activation = nn.Softmax(dim=1) # dim表示应用softmax的维度， dim=1表示对第二维， 也就是列应用softmax
        self.loss = nn.CrossEntropyLoss() # 交叉熵损失函数
        
    
    def forward(self, x:torch.Tensor, y:torch.Tensor=None):
        '''
        x: 我们输入的样本矩阵
        y: 样本的真实标签， 如果y给出了，我们需要返回loss， 如果
            没给出，我们直接返回模型预测的y_pred
        '''
        
        # 得到线性层的输出
        linear_list = self.linear_list
        
        for i, ll in enumerate(linear_list):
            x = ll(x)
            if i!=len(linear_list)-1:
                x =self.hidden_activation()(x) # 过一下线性层的激活函数
        
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
    
    
 # ================================ 新加代码: A* search NN's optimal hyper-params =========================   

# 优先队列
import heapq

class Node:
    def __init__(self, layers, units, activation, performance:float=0):
        # hidden layer numbers
        self.layers = layers
        # hidden units numbers
        self.units = units
        # nn.Relu ... 
        self.activation:nn.Module = activation
        # avg(f1+auc)
        self.performance:float = performance
        # 父节点指针
        self.parent:Node = None
        # 从开始在当前节点的路径上的节点数 (包括当前)
        self.path_node_num: int = None
        
        # cost(start, current)
        self.cost:float=0

    def __lt__(self, other):
        return self.performance < other.performance
    
    

def A_star_search(initial_node:Node, train_x:torch.Tensor, train_y:torch.Tensor, test_x:torch.Tensor, test_y:torch.Tensor, test_y_label:np.ndarray, encoder:LabelEncoder):
    '''
    A* 算法， 用来寻找使得神经网络分类性能最高的超参数：线性层的层数， 每层hidden unit的数量， 激活函数的种类
    '''
    queue = []
    heapq.heappush(queue, initial_node)

    while queue:
        current_node:Node = heapq.heappop(queue)
        if is_goal(current_node):
            print(f'A*最优的参数是：layers:{current_node.layers},  hiddenUnits:{current_node.units},  activation:{current_node.activation}\n')
            print(f'参数最优时的performance (f1 + auc + pre + recall + acc)/5 = {1/current_node.performance}')
            return current_node

        neighbors = get_neighbors(current_node)
        for neighbor in neighbors:
            neighbor.performance = evaluate_node(node = neighbor, start=initial_node, goal = 0.96, 
                                                 train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_y_label=test_y_label, encoder=encoder)
            heapq.heappush(queue, neighbor)

    return None



def get_neighbors(node:Node):
    # Generate neighbors by changing one hyperparameter at a time
    neighbors = []
    
    # generate neighbor by changing NN's layer number
    # 减少、或增加层数的节点都可以被当做邻居
    if node.layers > 1:
        neighbors.append(Node(node.layers - 1, node.units, node.activation, 0))
    neighbors.append(Node(layers=node.layers+1, units=node.units, activation=node.activation, performance=0))
    
    
    # Generate neighbors by changing the number of units
    if node.units > 1:  
        neighbors.append(Node(node.layers, node.units - 1, node.activation, 0))
    neighbors.append(Node(node.layers, node.units + 1, node.activation, 0))
    
    
    # Generate neighbors by changing the activation function
    activation_functions = [nn.ReLU, nn.Sigmoid, nn.Tanh]
    current_activation_index = activation_functions.index(node.activation)
    next_activation_index = (current_activation_index + 1) % len(activation_functions)
    neighbors.append(Node(node.layers, node.units, activation_functions[next_activation_index], 0))
    
    return neighbors
    
'''
以下这段是周三pre完后加的
'''
def evaluate_node(node:Node, start:Node, goal:float,
                  train_x:torch.Tensor, train_y:torch.Tensor, 
                  test_x:torch.Tensor, test_y:torch.Tensor, test_y_label:np.ndarray, encoder:LabelEncoder):
    # Train and evaluate a neural network with the given hyperparameters
    # g = 平均性能的倒数
    g_value = cost(start, node, train_x, train_y, test_x, test_y, test_y_label, encoder)
    # h = 平均性能的倒数
    h_value = heuristic(node, goal)
    # f是前面两个平均性能的倒数
    f_value = 1/((1/g_value + 1/h_value)/2)
    
    return f_value

def heuristic(node:Node, goal:float):
    
    if node.performance ==0:
        return 1/goal
    # 取当前节点性能与目标节点性能平均的倒数
    return 1/((1/node.performance+goal)/2)

'''
以下这段是周三pre完后加的
'''
def cost(start:Node, node:Node, train_x:torch.Tensor, train_y:torch.Tensor, 
         test_x:torch.Tensor, test_y:torch.Tensor, test_y_label:np.ndarray, encoder:LabelEncoder):
    
    model = TorchModel(
        input_size=train_x.shape[1], layers=node.layers, units=node.units, hidden_activation=node.activation)
    
    print(f'当前节点上， 神经网络的超参数是: layers = {node.layers}, hidden units = {node.units}, activation = {node.activation}')
    
    trained_model = train_model(model, train_x, train_y)
    
    # 获得预测值
    test_x=test_x.detach()
    y_pred:torch.Tensor= trained_model(test_x)
    y_pred=y_pred.detach() # 2947 x 6
    y_pred = torch.argmax(y_pred, dim=1).detach()
    # 转为string标签   
    y_pred_label = encoder.inverse_transform(y_pred)
    
    # 计算f1
    f1 = f1_score(test_y_label, y_pred_label, average='macro')
    
    # 计算auc
    auc = roc_auc_score(label_binarize(test_y, classes=[0,1,2,3,4,5]), label_binarize(y_pred, classes=[0,1,2,3,4,5]), multi_class='ovr')

    
    accuracy = accuracy_score(test_y, y_pred)
    
    precision = precision_score(test_y, y_pred, average= 'macro')
    
    recall = recall_score(test_y, y_pred, average= 'macro')


    print(f'当前f1 = {f1}, 当前auc = {auc}, 当前precision = {precision}, 当前recall = {recall}, 当前accuracy = {accuracy}')
    
    # 当前模型的性能
    node_cost = (f1 + auc + accuracy + precision + recall)/5
    
    # 当前真实代价(cost)和父节点代价做一个平均 ==> 模拟从原点到当前节点的代价
    # 取倒数是因为我们比的是谁的代价更小
    if node.parent is not None:
        avg_cost = ((1/node.parent.cost) + node_cost)/2
    else:
        avg_cost = node_cost
    
    # 节点的最终cost == 节点所在的<start, node>路径上的平均性能
    # 为什么要用倒数， 因为cost比较的是谁的值更小
    node.cost = 1/avg_cost
    
    return node.cost

def is_goal(node:Node):
    # Check if the performance of the node is good enough
    
    '''
    我们这里将f1-score 和 auc的平均值设为性能指标
    '''
    goal = 0.96
    
    if node.performance == 0:
        return False
    return (1/node.performance) >= goal


    








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
    
    # normalize every values in feature columns
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled:np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.fit_transform(X_test)
    
    
    # 将所有数据集全部转为Tensor
    X_train = torch.FloatTensor(X_train_scaled)
    Y_train = torch.FloatTensor(Y_train)
    X_test = torch.FloatTensor(X_test_scaled)
    Y_test = torch.FloatTensor(Y_test)
    
    # print(Y_train)
    
    
    # 准备画布
    plt.figure(figsize=(10, 8))
    
    # 使用饼图展示原始数据分布
    temp = train["Activity"].value_counts()
    df = pd.DataFrame({'labels': temp.index,
                    'values': temp.values
                    })

    labels = df['labels']
    sizes = df['values']
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','cyan','lightpink']
    plt.pie(sizes, colors=colors, shadow=True, startangle=90, labeldistance=1.2, autopct='%1.1f%%')
    plt.legend(labels, loc="best", ncol=2)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    
    
    print(f"X_train.shape：{X_train.shape}")
    print(f"Y_train.shape：{Y_train.shape}")
    print(f"X_test.shape：{X_test.shape}")
    print(f"Y_test.shape：{Y_test.shape}")
    print('--------------------------------')
    print(f"Number of feature = {X_train.shape[1]}")
    
    return  X_train, Y_train, X_test, Y_test, Y_test_label, le



def predict():
    pass


# 测试每轮(epoch)模型的准确率
def evaluate(model:TorchModel, test_x, test_y):
    model.eval()
    
    # 导入测试集
    # _, _, test_x, test_y = build_dataset()
    
    test_sample_size=len(test_x)
    correct, wrong = 0, 0 # 预测正确的个数， 预测错误的个数
    
    with torch.no_grad():
        # 预测值是一个one-hot的0-1向量，1所在的位置代表真实类别
        test_y_pred: torch.Tensor= model(test_x)
        
        # print('test_y_pred:\n {test_y_pred}')
        # print('test_y:\n {test_y}')
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

# def one_hot(y_pred):
#     '''
#     将概率分布矩阵转为0-1的二值矩阵
#     '''
#     one_hot = torch.zeros_like(y_pred)
    
#     i=0
#     for row in y_pred:
#         one_index = torch.argmax(row).item()
#         one_hot[i, one_index] =1
#         i+=1
        
#     return one_hot


def one_hot(y_pred:torch.Tensor):
    '''
    将一列标签向量 转为0-1的二值矩阵
    '''
    y_pred=y_pred.long()
    num_classes = (torch.max(y_pred) + 1).item()
    one_hot = torch.zeros((y_pred.shape[0], int(num_classes)))
    one_hot.scatter_(1, y_pred.unsqueeze(1), 1)
    return one_hot


def prob_to_one_hot(y_prob:torch.Tensor)-> torch.Tensor:
    '''
    
    将一个概率矩阵（类概率分布矩阵），转为0-1二值矩阵

    
    '''
    max_indexed = torch.argmax(y_prob, dim=1)
    one_hot = torch.zeros_like(y_prob)
    
    for i, j in enumerate(max_indexed):
        one_hot[i,j]=1
        
    return one_hot
    


def one_hot_to_single(y_pred):
    '''
    将one-hot编码转为单个分类值
    '''
    
    y_pred_label=[]
    
    for x in y_pred:
        y_pred_label.append(torch.argmax(x).item())
    
    return y_pred_label

def get_roc(test_y:torch.Tensor,test_y_pred:torch.Tensor):
    '''
    test_y: 测试集的标签向量
    test_y_pred: 测试集的标签对应的概率分布矩阵：shape(n, 6)
    
    一对多策略的含义
    它是一种多分类问题中常用的方法，它将多分类问题分解为多个二分类问题来解决。
    具体来说，对于一个拥有 N 个类别的多分类问题，一对多策略会为每个类别创建一个二分类模型，
    每个模型都将该类别视为正类，将其他所有类别视为负类。
    这样，我们可以得到 N 个二分类模型，每个模型都能够区分一个特定的类别与其他所有类别。
    
    根据这种策略， 我们就可以画出N个ROC图像
    
    '''
    
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning

    # 忽略 UndefinedMetricWarning, 用于忽略控制台的警告信息
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


    # 准备画布
    plt.figure(figsize=(10, 8))
    
    from sklearn.preprocessing import label_binarize


    # 为每个类别绘制ROC曲线
    for i in range(test_y_pred.shape[1]):
        # test_y_pred[:, i] 是一个一维数组，表示模型预测每个样本属于第 i 类的概率。
        # pos_label 参数在 roc_curve 函数中用于定义哪个类别被视为正类。
        # 将 test_y_bin[:, i], test_y_pred[:, i] 结合起来可以计算 TPR 和 FPR
        y=test_y.detach().numpy()
        y_prob = test_y_pred
        y_prob = y_prob[:, i].detach().numpy()
        

        
        
        fpr, tpr, thresholds = roc_curve(y, y_prob, pos_label=i)
        # 计算auc的值
        roc_auc = auc(fpr, tpr)
        # lw: linewidth
        plt.plot(fpr, tpr, lw = 2, label = 'ROC curve of class %d, AUC = %0.2f'%(i, roc_auc))

    # 添加对角线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # 设置图的其他属性
    # 将 x 轴的显示范围设置为从 0.0 到 1.0
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RUC for multi-class')
    plt.legend(loc="lower right")
    plt.show()
    


def k_fold_cross_validation(k):
    '''
    我们将训练集分为K折，其中K-1份组成训练集，剩下一份是测试集，轮流跑K次，最后取平均测试结果
    '''
    input_size = 561  # 输入向量维度
    learning_rate = 0.001  # 学习率  
    
    model = TorchModel(input_size=input_size)
    
    # 创建优化器
    optim = torch.optim.Adam(params = model.parameters(), lr=learning_rate)
    
    dataset:list[list] = shuffle(pd.read_csv('./train.csv'))
    
    dataset: pd.DataFrame = pd.DataFrame(dataset)
    
    # 分离特征和标签， 以及类别映射
    X_dataset = pd.DataFrame(dataset.drop(['Activity','subject'], axis=1))
    X_dataset = torch.tensor(X_dataset.values, dtype = torch.float32)
    
    Y_dataset_label = dataset.Activity.values.astype(object)
    labelEncoder = LabelEncoder()
    Y_dataset: np.ndarray =labelEncoder.fit_transform(Y_dataset_label)
    Y_dataset = torch.tensor(Y_dataset, dtype = torch.long)
    
    
    # dataset=np.array(dataset)
    
    # print(dataset)
    dataset_len = len(dataset)
    fold_size = dataset_len // k
    
    # 记录每一轮，测试集上的损失
    watch_loss= []
    
    # 在开始交叉验证之前保存模型的初始参数
    import copy
    initial_state_dict = copy.deepcopy(model.state_dict())
    
    for i in range(k): # k轮交叉验证
        
        # 在每次迭代开始时加载初始参数
        model.load_state_dict(copy.deepcopy(initial_state_dict))
            
        X_test = X_dataset[i*(fold_size):(i+1)*fold_size]
        Y_test = Y_dataset[i*(fold_size):(i+1)*fold_size]
        
        X_train = torch.cat((X_dataset[:i*fold_size], X_dataset[(i+1)*fold_size:]), dim=0)
        Y_train = torch.cat((Y_dataset[:i*fold_size], Y_dataset[(i+1)*fold_size:]), dim=0)
        # X_train = X_dataset[(i+1)*fold_size:]
        # Y_train = Y_dataset[(i+1)*fold_size:]
    
        loss = model(X_train, Y_train)
        loss.backward()  # 计算梯度
        optim.step() # 更新参数
        model.zero_grad() # 梯度归零， 每一个批次只能用该批次的损失函数来计算梯度
        
        # 测试集的误差
        loss_test = model(X_test, Y_test)
        watch_loss.append(loss_test.item())   
        
        print(f'{k}-fold round # {i+1}, loss = {loss_test.item():.2f}')
        # acc=evaluate(model, test_x, test_y)
        # log.append([acc,np.mean(watch_loss)])

    print(f'{k}-fold CV\'s average loss = {np.mean(watch_loss):.2f}')
    

    # 准备画布
    plt.figure(figsize=(10, 8))
    
    # 绘制损失的历史数据
    plt.plot(watch_loss)
    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()






'''
以下这段是周三pre完后加的
'''
def fine_tune():
    '''
    运行A*算法， 微调模型的超参数 （找到最佳的NN参数组合）
    '''
    
    import time
    start = time.time()
    
    # 创建训练集
    train_x, train_y, test_x, test_y, test_y_label, final_encoder = build_dataset()
    
    
    print(f"============= A* 预计要运行 10分钟 =========")
    
    # model = TorchModel(input_size=561,layers=4, units=10, hidden_activation=nn.ReLU())
    
    
    # final_model = train_model(model=model,train_x=train_x, train_y=train_y)
    
    initial_node = Node(layers =5, units=10, activation = nn.ReLU, performance=0)
    final_node:Node = A_star_search(initial_node, train_x, train_y, 
                  test_x, test_y, test_y_label, final_encoder)

    
    model1 = TorchModel(input_size=561, layers = final_node.layers, units=final_node.units,hidden_activation=final_node.activation)
    
    final_model = train_model(model1, train_x, train_y)
    
    end=time.time()
    
    print(f'A* search 总共运行了: {end-start} 秒')
    
    return final_model
    
    
def train_model(model:TorchModel, train_x:torch.Tensor, train_y:torch.Tensor):
    # 配置参数
    epoch_num = 5  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = len(train_x)  # 每轮训练总共训练的样本总数
    input_size = train_x.shape[1]  # 输入向量维度
    learning_rate = 0.001  # 学习率  

    # parameters: 接收模型参数
    optim = torch.optim.Adam(params = model.parameters(), lr=learning_rate)
    
    # 损失日志
    log = []
    
    # 创建训练集
    # train_x, train_y, test_x, test_y, test_y_label, final_encoder = build_dataset()
    
    final_encoder: LabelEncoder
    
    print("==========开始模型训练=============")
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
            
            # print(f'y_batch:\n{y_batch}')
            # print(len(y_batch))
            
            loss = model(x_batch, y_batch)
            
            loss.backward()  # 计算梯度
            optim.step() # 更新参数
            
            model.zero_grad() # 梯度归零， 每一个批次只能用该批次的损失函数来计算梯度
            
            watch_loss.append(loss.item())   
        
        print(f'epoch #{epoch+1}, average loss = {np.mean(watch_loss):.2f}')
        # accuracy=evaluate(model, test_x, test_y)
 


    
    # # 显示分类完成后，模型的分类性能
    # test_y_pred = model(test_x)
    # print(f'test_y_pred:\n {test_y_pred}')
    

    return model
    


# 执行训练任务
def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
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
    train_x, train_y, test_x, test_y, test_y_label, final_encoder = build_dataset()
    
    final_encoder: LabelEncoder
    
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
            
            # print(f'y_batch:\n{y_batch}')
            # print(len(y_batch))
            
            loss = model(x_batch, y_batch)
            
            loss.backward()  # 计算梯度
            optim.step() # 更新参数
            
            model.zero_grad() # 梯度归零， 每一个批次只能用该批次的损失函数来计算梯度
            
            watch_loss.append(loss.item())   
        
        print(f'epoch #{epoch+1}, average loss = {np.mean(watch_loss):.2f}')
        acc=evaluate(model, test_x, test_y)
        log.append([acc,np.mean(watch_loss)])
    
    # 保存模型到本地文件
    # torch.save(model.state_dict(), "classifier1.pt")
    
    # 画出性能曲线
    print(log)
    

    # 准备画布
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(log)), [x[0] for x in log], label = 'accuracy')
    plt.plot(range(len(log)),[x[1] for x in log], label = 'loss')
    plt.legend()
    plt.show()
    
    # 显示分类完成后，模型的分类性能
    test_y_pred = model(test_x)
    print(f'test_y_pred:\n {test_y_pred}')
    
    test_y_pred_label = one_hot_to_single(test_y_pred)
    # print(f"test_y_pred_label = {test_y_pred_label}")
    test_y_pred_label=final_encoder.inverse_transform(test_y_pred_label)
    print("\n=========== confusion matrix ==============")
    print(confusion_matrix(test_y_label,test_y_pred_label))
    
    
    print("\n=========== classification report ==============")
    print(classification_report(test_y_label,test_y_pred_label))
    
    
    print("\n=========== ROC ==============")
    
    get_roc(test_y=test_y, test_y_pred=test_y_pred)
    

    
    print("\n============= 10-fold cross validation ============")
    k_fold_cross_validation(50)
    
    return model
    


def print_torch():
    model = main()
    return model


def print_torch_optimized():
    model = fine_tune()
    return model

if __name__ == '__main__':
    # build_dataset()
    # main()
    # k_fold_cross_validation(50)
    
    # 创建训练集
    train_x, train_y, test_x, test_y, test_y_label, final_encoder = build_dataset()
    
    
    # model = TorchModel(input_size=561,layers=4, units=10, hidden_activation=nn.ReLU())
    
    
    # final_model = train_model(model=model,train_x=train_x, train_y=train_y)
    
    initial_node = Node(layers =5, units=10, activation = nn.ReLU, performance=0)
    final_node:Node = A_star_search(initial_node, train_x, train_y, 
                  test_x, test_y, test_y_label, final_encoder)

    
    model1 = TorchModel(input_size=561, layers = final_node.layers, units=final_node.units,hidden_activation=final_node.activation)
    
    final_model = train_model(model1, train_x, train_y)
    
