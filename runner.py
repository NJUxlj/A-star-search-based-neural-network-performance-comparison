# This is the master file of our project

# Just run it !

import activity_classifier_torch

from activity_classifier_sklearn import *
from activity_classifier_torch import *
from activity_classifier_transformer import *

import time


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
    
    # print(f'Dimension of the train set: {X_train.shape}')
    # print(f'Dimension of the test set: {X_test.shape}')
    # print(f'Number of features = {X_train.shape[1]}')
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
    

    
    return  X_train_scaled, Y_train, X_test_scaled, Y_test



def compare_models(model1:TorchModel, model2:SVMModel):

    
    
    
    '''
    比较3中不同模型的分类性能
    
    model1 和 model2 必须是已经训练好的模型
    '''
    
    # 导入数据集
    X_train_scaled, Y_train, X_test_scaled, Y_test = build_dataset()
    
    
    X_test_scaled_model1 = torch.Tensor(X_test_scaled)
    Y_test_model1 = torch.Tensor(Y_test)
    
    # 使用TorchModel进行预测
    model1_pred = model1(X_test_scaled_model1)
    model1_accuracy = accuracy_score(Y_test_model1, model1_pred)

    # 使用SVMModel进行预测
    model2_pred = model2.predict(X_test_scaled)
    model2_accuracy = accuracy_score(Y_test, model2_pred)

    # 创建条形图
    plt.bar(['NeuralNetwork', 'SVM'], [model1_accuracy, model2_accuracy])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.show()
    


if __name__ == '__main__':
    '''
    
    '''
    
    # run the 1st model and check performance
    model1 = print_torch()
    
    time.sleep(1)
    
    print(f"========== Next we are going to show the second model's performance ============")
    
    # run the second model
    model2 = print_svm()
    
    
    
    compare_models(model1, model2)