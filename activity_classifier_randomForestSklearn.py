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

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# from activity_classifier_decisionTree import *



class RandomForestModel():
    def __init__(self):
        # 代表森林中树木的数量， 是一个超参数， 会影响模型性能
        # 作用：
            # 1. 提高模型泛化能力
            # 2. 控制模型复杂度
            # 3. 取值：100-1000
            # 4. 如何找到最优的 n_estimators:交叉验证
        self.n_estimators = 100
        self.random_state = 42
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

    
    def modify_params(self, n_estimators, random_state):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)


        print(f'You have already change the model parameters to:\n n_estimators={self.n_estimators}\nrandom_state={self.random_state}')

    
    def train(self,X_train_scaled,Y_train):
        final_model = self.model.fit(X_train_scaled,Y_train)
        
        return self


    def predict(self, X_test_scaled):
        y_pred = self.model.predict(X_test_scaled)
        return y_pred
    
    
    
    
# 测试每轮(epoch)模型的准确率
def evaluate(model:RandomForestModel, test_x, test_y):

    
    test_sample_size=len(test_x)
    correct, wrong = 0, 0 # 预测正确的个数， 预测错误的个数
    
    # 预测值是一个向量，每个值是一个class-integer
    test_y_pred= model.predict(X_test_scaled=test_x)
    

    for y, y_pred in zip(test_y, test_y_pred):
        
        if y_pred == y: 
            correct+=1
        else:
            wrong+=1
    print(f'分类的正确率为：{correct/(correct+wrong)}')
    return correct/(correct+wrong), test_y_pred





    
def build_dataset()->Union[typing.List[np.ndarray], StandardScaler]:
    
    '''
    创建训练集和测试集
    '''
    train: pd.DataFrame = pd.DataFrame(shuffle(pd.read_csv('./train.csv')))
    test: pd.DataFrame = pd.DataFrame(shuffle(pd.read_csv('./test.csv')))
    
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
    
    
    # 将数据集中的特征列和标签列分离开来
    X_train = pd.DataFrame(train.drop(['Activity','subject'],axis=1))
    # values：获取DataFrame列的值
    # astype(object): 将列中的值转为object类型
    Y_train_label = train.Activity.values.astype(object)
    
    
    X_test = pd.DataFrame(test.drop(['Activity','subject'],axis=1))
    Y_test_label = test.Activity.values.astype(object)
    
    
    # dataset最后一列类标签转为数字0-5
    encoder = LabelEncoder()
    encoder.fit(Y_train_label)
    Y_train = encoder.transform(Y_train_label)
    
    encoder.fit(Y_test_label)
    # get am array that each ele is a class-mapping-number [0-5]
    Y_test = encoder.transform(Y_test_label)

    # 里面存了针对Y_test_label的编码器
    final_encoder = encoder
    print(f'Dimension of the train set: {X_train.shape}')
    print(f'Dimension of the test set: {X_test.shape}')
    print(f'Number of features = {X_train.shape[1]}')
    # print(f'Name of all features: \n{X_train.columns.values}')

    # normalize every values in feature columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    # # 将所有数据集全部转为Tensor
    # X_train = torch.FloatTensor(X_train.to_numpy())
    # Y_train = torch.FloatTensor(Y_train)
    # X_test = torch.FloatTensor(X_test.to_numpy())
    # Y_test = torch.FloatTensor(Y_test)
    

    
    return  X_train_scaled, Y_train, X_test_scaled, Y_test, Y_test_label, final_encoder





def get_roc(test_y: np.ndarray, test_y_scores:np.ndarray):
    # Compute ROC curve and ROC area for each class

    import warnings
    from sklearn.exceptions import UndefinedMetricWarning

    # 忽略 UndefinedMetricWarning, 用于忽略控制台的警告信息
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


    # 准备画布
    plt.figure(figsize=(10, 8))

    for i in range(test_y_scores.shape[1]):
        y=test_y
        y_prob =test_y_scores[:,i]
        fpr, tpr, _ = roc_curve(y, y_prob, pos_label=i)  # one-to-M policy for binary classification, and you're interested in class i

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='ROC curve of class %d, AUC = %0.2f' % (i, roc_auc))
        
    # Plot diagnal
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()



def cross_validation():
    '''
    利用交叉验证来选择随机森林的参数
    
    '''
    
    
    
    
def main():
    X_train_scaled, Y_train, X_test_scaled, Y_test, Y_test_label, final_encoder=build_dataset()

    
    model = RandomForestModel()
    
    final_model = model.train(X_train_scaled,Y_train)
    
    _, test_y_pred = evaluate(final_model, X_test_scaled, Y_test)
    
    final_encoder: LabelEncoder
    Y_pred_label=final_encoder.inverse_transform(test_y_pred)
    
    # 混淆矩阵
    print(confusion_matrix(Y_test_label, Y_pred_label))
    
    print('\n\n')
    
    # 展示分类准确度指标报告
    print(classification_report(Y_test_label, Y_pred_label))
    
    return final_model



def print_randomForestSklearn():
    final_model = main()
    return final_model

if __name__ == '__main__':
    main()