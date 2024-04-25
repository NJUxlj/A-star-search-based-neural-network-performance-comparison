from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from typing import Union
import typing
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

# 导入交叉验证和参数最优化的包
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc



import numpy as np
import pandas as pd


'''
这整个文件都是周三pre完后加的
'''

def calculate_entropy(y):
    """
    计算给定标签的熵
    """
    
    # 找出数组y中的所有唯一值，并将这些唯一值赋值给class_labels
    class_labels = np.unique(y)
    entropy = 0
    for cls in class_labels:
        # y[y == cls]是一个布尔索引，它会返回一个新的数组，这个新的数组只包含y中等于cls的元素。
        p = len(y[y == cls]) / len(y)
        entropy -= p * np.log2(p)
    return entropy

def calculate_information_gain(X, y, feature_index):
    """
    计算某一个特征的information gain
    """
    # 总熵
    total_entropy = calculate_entropy(y)
    
    # 计算权重熵
    
    values, counts = np.unique(X[:, feature_index], return_counts=True) # 获取特征feature_index所有唯一值及其出现次数
    weighted_entropy = 0
    for v, c in zip(values, counts):
        sub_y = y[X[:, feature_index] == v] # 获取特征值为v的所有样本的标签, 并组成一个子集
        # 计算子集的熵
        entropy = calculate_entropy(sub_y)
        # 计算这些样本的加权熵，并累加到weighted_entropy上。权重是这些样本的比例，即c / len(X)。
        weighted_entropy += (c / len(X)) * entropy
    
    '''
    weighted_entropy就是特征feature_index的加权熵。
    这个值反映了根据该特征划分数据集后，各子集的熵的加权平均。
    在决策树算法中，我们通常会选择加权熵最小的特征作为划分特征，因为这样可以获取最大的信息增益。
    
    '''
    
    
    # 信息增益
    information_gain = total_entropy - weighted_entropy
    return information_gain

def best_feature_to_split(X, y):
    """找到最佳分割特征的索引"""
    n_features = X.shape[1]
    information_gains = [calculate_information_gain(X, y, i) for i in range(n_features)]
    return np.argmax(information_gains)

def build_tree(X, y, depth=0, max_depth=10):
    """递归构建决策树"""
    if len(np.unique(y)) == 1 or len(y) == 0:
        # 如果所有输出相同，或者没有数据，返回结果
        return np.unique(y)[0] if len(y) > 0 else None
    
    if depth == max_depth:
        # 达到最大深度，返回多数类
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    # 选择最佳分割特征
    feature_index = best_feature_to_split(X, y)
    values = np.unique(X[:, feature_index])
    tree = {}
    tree[feature_index] = {}
    
    # 为每个特征值构建子树
    for value in values:
        sub_X = X[X[:, feature_index] == value]
        sub_y = y[X[:, feature_index] == value]
        tree[feature_index][value] = build_tree(sub_X, sub_y, depth + 1, max_depth)
    
    return tree

# 示例数据加载和模型训练
# 假设 data 是已经加载的 DataFrame，包含 251 个特征和 1 个标签列
data = pd.read_csv("your_dataset.csv")
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 标签

# 构建决策树
decision_tree = build_tree(X, y)

# 使用决策树进行预测的函数（这里仅提供框架，需要具体实现）
def predict(tree, x):
    """在决策树上预测单个样本"""
    for feature_index, branches in tree.items():
        value = x[feature_index]
        if value in branches:
            subtree = branches[value]
            if type(subtree) is dict:
                return predict(subtree, x)
            else:
                return subtree
        else:
            return None  # 无法处理的值




if __name__ == '__main__':
    # 预测示例
    x_new = X[0]  # 假设我们使用第一个样本进行预测
    print(predict(decision_tree, x_new))
