# This is the master file of our project

# Just run it !

import activity_classifier_torch

from activity_classifier_sklearn import *
from activity_classifier_torch import *
from activity_classifier_transformer import *
from activity_classifier_randomForestSklearn import *

import time
from typing import Union
import typing

from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize



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
    
    
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    # # 将所有数据集全部转为Tensor
    # X_train = torch.FloatTensor(X_train.to_numpy())
    # Y_train = torch.FloatTensor(Y_train)
    # X_test = torch.FloatTensor(X_test.to_numpy())
    # Y_test = torch.FloatTensor(Y_test)
    

    
    return  X_train_scaled, Y_train, X_test_scaled, Y_test



def compare_models(**kwargs:Union[TorchModel, SVMModel, RandomForestModel,list]):

    '''
    比较3中不同模型的分类性能
    
    model1 和 model2 必须是已经训练好的模型
    '''
    
    # 导入数据集
    X_train_scaled, Y_train, X_test_scaled, Y_test = build_dataset()
    
    
    X_test_scaled_model1 = torch.Tensor(X_test_scaled).detach()
    Y_test_model1 = torch.Tensor(Y_test).detach()
    
    # 使用TorchModel进行预测
    # model1_pred = model1(X_test_scaled_model1).detach()
    model1_pred = torch.max(model1(X_test_scaled_model1), 1)[1].detach()
    model1_accuracy = accuracy_score(Y_test_model1, model1_pred)
    model1_precision = precision_score(Y_test_model1, model1_pred, average='macro')
    model1_recall = recall_score(Y_test_model1, model1_pred, average='macro')
    model1_f1 = f1_score(Y_test_model1, model1_pred, average='macro')
    model1_roc_auc = roc_auc_score(label_binarize(Y_test_model1, classes=[0,1,2,3,4,5]), label_binarize(model1_pred, classes=[0,1,2,3,4,5]), multi_class='ovr')


    # 使用SVMModel进行预测
    model2_pred = model2.predict(X_test_scaled)
    model2_accuracy = accuracy_score(Y_test, model2_pred)
    model2_precision = precision_score(Y_test, model2_pred, average='macro')
    model2_recall = recall_score(Y_test, model2_pred, average='macro')
    model2_f1 = f1_score(Y_test, model2_pred, average='macro')
    model2_roc_auc = roc_auc_score(label_binarize(Y_test, classes=[0,1,2,3,4,5]), label_binarize(model2_pred, classes=[0,1,2,3,4,5]), multi_class='ovr')
    
    
    # 使用RandomForest进行预测
    model3_pred = model3.predict(X_test_scaled)
    model3_accuracy = accuracy_score(Y_test, model3_pred)
    model3_precision = precision_score(Y_test, model3_pred, average='macro')
    model3_recall = recall_score(Y_test, model3_pred, average='macro')
    model3_f1 = f1_score(Y_test, model3_pred, average='macro')
    model3_roc_auc = roc_auc_score(label_binarize(Y_test, classes=[0,1,2,3,4,5]), label_binarize(model3_pred, classes=[0,1,2,3,4,5]), multi_class='ovr')
    
    
    
    # Bert的性能数据已经由Bert所在的文件返回
    
    
    
    # # 使用优化后的TorchModel进行预测
    X_test_scaled_model5 = torch.Tensor(X_test_scaled).detach()
    Y_test_model5 = torch.Tensor(Y_test).detach()
    
    # 使用TorchModel进行预测
    # model1_pred = model1(X_test_scaled_model1).detach()
    model5_pred = torch.max(model5(X_test_scaled_model5), 1)[1].detach()
    model5_accuracy = accuracy_score(Y_test_model5, model5_pred)
    model5_precision = precision_score(Y_test_model5, model5_pred, average='macro')
    model5_recall = recall_score(Y_test_model5, model5_pred, average='macro')
    model5_f1 = f1_score(Y_test_model5, model5_pred, average='macro')
    model5_roc_auc = roc_auc_score(label_binarize(Y_test_model5, classes=[0,1,2,3,4,5]), label_binarize(model5_pred, classes=[0,1,2,3,4,5]), multi_class='ovr')

    
    
    
    
    
    # 有其他模型， 往下继续加就行

    # 创建条形图
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC']
    model1_scores = [model1_accuracy, model1_precision, model1_recall, model1_f1, model1_roc_auc]
    model2_scores = [model2_accuracy, model2_precision, model2_recall, model2_f1, model2_roc_auc]
    model3_scores = [model3_accuracy, model3_precision, model3_recall, model3_f1, model3_roc_auc]
    # model4_scores= model4_metrics_list
    model4_scores = [0.71, 0.76, 0.71, 0.73, 0.71]
    model5_scores = [model5_accuracy, model5_precision, model5_recall, model5_f1, model5_roc_auc]
    
    
    x = np.arange(len(metrics))  # the label locations
    width = 0.25  # the width of the bars
    
    # fig: 图形窗口
    # ax: sub-plot
    fig, ax = plt.subplots()
    # 条形子图
    rects1 = ax.bar(x + 1*width/5, model1_scores, width, label='TorchModel')
    rects2 = ax.bar(x + 2*width/5, model2_scores, width, label='SVMModel')
    rects3 = ax.bar(x + 3*width/5, model3_scores, width, label='RandomForestModel')
    rects4 = ax.bar(x + 4*width/5, model4_scores, width, label='Bert')
    rects5 = ax.bar(x + 5*width/5, model5_scores, width, label='optimized TorchModel')
    
    
    
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Performance Scores')
    ax.set_title('Classification Performance between Models')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    fig.tight_layout()  

    # plt.bar(['NeuralNetwork', 'SVM'], [model1_accuracy, model2_accuracy])
    # plt.xlabel('Model')
    # plt.ylabel('Accuracy')
    # plt.title('Model Comparison')
    
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
    
    
    
    model3 =print_randomForestSklearn()
    
    
    # model4, model4_metrics_list = print_transformer()
    
    model5 = print_torch_optimized()
    
    
    
    compare_models(model1 = model1, model2 = model2, model3 = model3, model5 =model5, model4 = None, model4_metrics_list=None)