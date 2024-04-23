import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
from typing import Union
import typing


# sklearn related packages
from sklearn.utils import shuffle
# 导入标签编码器
from sklearn.preprocessing import LabelEncoder
# 导入混淆矩阵
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


from activity_classifier_decisionTree import *



def bootstrap_sample(X, y):
    """
    生成一个自助采样集
    """
    index = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
    return X[index], y[index]

def feature_sampling(X, ratio=0.3):
    """
    随机选择部分特征
    """
    n_features = int(X.shape[1] * ratio)
    indices = np.random.choice(X.shape[1], n_features, replace=False)
    return X[:, indices], indices

def build_forest(X, y, n_trees=10, max_depth=10, feature_ratio=0.3):
    """构建随机森林"""
    forest = []
    for _ in range(n_trees):
        # 生成自助采样集
        X_sample, y_sample = bootstrap_sample(X, y)
        # 随机选择特征
        X_sample, feature_indices = feature_sampling(X_sample, ratio=feature_ratio)
        # 构建决策树
        tree = build_tree(X_sample, y_sample, max_depth=max_depth)
        forest.append((tree, feature_indices))
    return forest

def forest_predict(forest, x):
    """
    使用随机森林进行预测
    """
    predictions = []
    for tree, features in forest:
        x_sub = x[features]
        prediction = predict(tree, x_sub)
        predictions.append(prediction)
    # 返回多数票决策
    return max(set(predictions), key=predictions.count)




def main():
    # 加载数据
    data = pd.read_csv("your_dataset.csv")
    X = data.iloc[:, :-1].values  # 特征
    y = data.iloc[:, -1].values   # 标签

    # 构建随机森林
    random_forest = build_forest(X, y, n_trees=100, max_depth=10, feature_ratio=0.3)

    # 使用随机森林进行预测
    x_new = X[0]  # 假设我们使用第一个样本进行预测
    print(forest_predict(random_forest, x_new))





