import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# %matplotlib inline

import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score

# 导入交叉验证和参数最优化的包
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

import torch

# 核函数的作用：
'''
 核函数可以被看作是: 在高维空间中计算两个向量的内积，而无需显式地将向量映射到这个高维空间。
                    这允许算法能够在高维空间中找到线性决策边界，从而解决原始空间中的非线性问题。
                    
    简单点：计算高维空间的内积，比如 K(X_i, X_j) = (X_i*X_j+C)^h
'''

class SVMModel:
    def __init__(self):
        # kernel function list
        # sigma：控制高斯核函数的宽度， sigma越小， 那么函数值随两点间（X_i, X_j)距离的增大而减小地越快
        # gamma: gamma = 1/(2*sigma^2)
        # C: 在多项式核函数中，确保了即使在原始特征全部为零时，也能有非零的核函数输出
        self.kernel=[
                    {
                        'kernel': ['rbf'], 
                        'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100, 1000]
                     },
                    {
                        'kernel': ['linear'], 
                        'C': [1, 10, 100, 1000]
                        }
                    ]
        # build a primary SVM model
        # SVC: SVM classifier
        # cv: number of folds in cross-validation
        self.model = GridSearchCV(SVC(probability=True),self.kernel, cv = 5)
    
    
    
    def set_kernel(self, kernel):
        '''
        设置核函数列表
        '''
        self.kernel = kernel
        
    def set_kernel_params(self, kernel_name, gamma, C):
        '''
        设置核函数的参数，比如：
        1. 多项式核的 h和c
        2. rbf和的sigma， 8它控制着核函数的宽度， 影响模型的灵敏度
        '''
        kernel_index = -1
        
        # search kernel function in the kernel list
        kernel_list = self.kernel
        for i, k_dict in enumerate(kernel_list):
            if kernel_list['kernel'] == kernel_name:
                kernel_index = i

        if kernel_index ==-1:
            return False
        else:
            kernel_list[kernel_index][kernel_name]['gamma'] = gamma
            kernel_list[kernel_index][kernel_name]['C'] = C
            return True
             
    
    def forward(self, x, y= None):
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

        print(f'Best C: {self.model.best_estimator_.C}') 
        print(f'Best Kernel: {self.model.best_estimator_.kernel}')
        print(f'Best Gamma: {self.model.best_estimator_.gamma}')
        
        final_model = self.model.best_estimator_
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




def main():
    '''
    展示模型指标
    '''

    X_train_scaled, Y_train, X_test_scaled, Y_test, Y_test_label, encoder = build_dataset()
    
    encoder: LabelEncoder
    Y_test=np.array(Y_test)
    
    svm= SVMModel()
    final_model:SVC = svm.evaluate(X_train_scaled,Y_train)
    
    # Y_pred:np.ndarray = final_model.predict(X_test_scaled)
    
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
    
    print("Training set score for SVM: %f" % final_model.score(X_train_scaled , Y_train))
    print("Testing  set score for SVM: %f" % final_model.score(X_test_scaled  , Y_test ))
    
    
    
    print(" ====================== ROC =============================")
    
    get_roc(Y_test, Y_pred_prob)
    
    return final_model

def print_svm():
    final_model = main()
    return final_model






if __name__ == '__main__':
    # build_dataset()
    main()