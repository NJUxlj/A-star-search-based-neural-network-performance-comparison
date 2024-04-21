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


class DecisionTreeModel:
    def __init__(self):
        # Decision tree parameters
        self.params = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }
        # build a primary Decision Tree model
        self.model = GridSearchCV(DecisionTreeClassifier(), self.params, cv=5)

    def forward(self, x, y=None):
        '''
        generate model's prediction
        '''

    def __fit(self, X_train_scaled, Y_train):
        '''
        fit the training data
        '''
        self.model.fit(X_train_scaled, Y_train)

    def evaluate(self, X_train_scaled, Y_train):
        print(" =======  It needs about 5 minutes, Please be patient ~~~ =====================")
        self.__fit(X_train_scaled, Y_train)
        print(f'Best score for training data:{self.model.best_score_}') 

        print(f'Best max_depth: {self.model.best_estimator_.max_depth}') 
        print(f'Best min_samples_split: {self.model.best_estimator_.min_samples_split}')
        print(f'Best min_samples_leaf: {self.model.best_estimator_.min_samples_leaf}')

        final_model = self.model.best_estimator_
        return final_model
    
    
    
    
    
def main():
    '''
    展示模型指标
    '''

    X_train_scaled, Y_train, X_test_scaled, Y_test, Y_test_label, encoder = build_dataset()
    
    encoder: LabelEncoder
    Y_test=np.array(Y_test)
    
    dt = DecisionTreeModel()
    final_model:DecisionTreeClassifier = dt.evaluate(X_train_scaled,Y_train)
    
    # get class probability scores
    Y_pred_prob:np.ndarray = final_model.predict_proba(X_test_scaled)
    
    print(f'Y_pred_prob.shape = {Y_pred_prob.shape}')
    Y_pred=np.argmax(Y_pred_prob, axis=1)

    # 0-5 的integer classes transfer to string classes
    Y_pred_label = list(encoder.inverse_transform(Y_pred))

    # 混淆矩阵
    print(confusion_matrix(Y_test_label, Y_pred_label))
    
    print('\n\n')
    
    # 展示分类准确度指标报告
    print(classification_report(Y_test_label, Y_pred_label))
    
    print("Training set score for Decision Tree: %f" % final_model.score(X_train_scaled , Y_train))
    print("Testing  set score for Decision Tree: %f" % final_model.score(X_test_scaled  , Y_test ))
    
    print(" ====================== ROC =============================")
    
    get_roc(Y_test, Y_pred_prob)
    
    return final_model




def build_dataset()->pd.DataFrame:
    
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




if __name__ == '__main__':
    main()