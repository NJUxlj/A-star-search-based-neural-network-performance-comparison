import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt 

# BertTokenizer: divide a text into many tokens
# BertForSequenceClassification: classify a text into certain class
# AdamW: a optimizer
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score

import numpy as np
from typing import Union
import typing






'''
transformer 可以用来做文本分类任务， 但是不能做普通的多分类任务， 因此我们需要做出一些修改


主要思想： 
1. 将原始训练数据中，每个样本的251个浮点型特征拼接成一条单一的文本
2. 将处理过后的数据集喂给transformer，让他为每个样本预测一个类
3. 记录分类性能指标，
4. 结束

'''

# 定义一些示例数据
# texts = ["政府今日宣布了新的经济政策", "球队在昨晚的比赛中表现出色", "股市今日大幅上涨"]
# labels = [1, 2, 1]  # 假设1代表经济，2代表体育

'''
NewsDataset: 自定义的数据集类，继承自PyTorch的Dataset类。它接受文本、标签和分词器作为输入，并处理文本的编码。
encode_plus: BERT分词器的方法，用于将文本转换为模型可以理解的格式，包括添加特殊令牌、截断、填充等。
'''

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        # 将文本转化为模型能够理解的格式， 比如tokens
        self.tokenizer = tokenizer
        self.encoder = LabelEncoder().fit(self.labels)
        self.labels = self.encoder.transform(self.labels)

    def __len__(self):
        '''
        return the length of the text
        '''
        return len(self.texts)

    def __getitem__(self, idx):
        '''
        :idx: 索引
        
        :return 
        '''
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 将文本分割为令牌
        # 添加特殊的开始和结束令牌
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            # 不足64，则自动填充
            max_length=64,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        '''
        'input_ids': 这是文本经过分词器处理后的结果，
                        每个词都被转换成了一个唯一的ID。这些ID是模型的输入，模型会用它们来查找词嵌入。
        
        'attention_mask': 这是一个与输入ID (input_ids) 相同长度的二进制向量，
                        用于指示哪些词是实际的词，哪些词是填充词。
                        例如，如果输入序列的长度小于64，那么剩余的位置会被填充为0，
                        对应的attention mask也会被设置为0。模型会忽略mask为0的词。
        
        'labels': 这是文本的标签，也就是我们希望模型预测的目标。
        '''
        
        
        return {
            # flatten: transfer multi-dimension vector into one-dimension vector

                    # 将input_ids张量从形状(batch_size, sequence_length)
                    # 转换为形状(batch_size * sequence_length,)
                    
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }


def evaluate(model:BertForSequenceClassification, test_labels:np.ndarray):
    '''
    评估模型
    '''
    model.eval()
    predictions = []
    with torch.no_grad(): # 使用torch.no_grad()来禁用梯度计算，从而减少内存消耗并加速计算
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())

    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy}")
    


def test_bert():
    # from transformers import pipeline
    # unmasker = pipeline('fill-mask', model='bert-base-uncased')
    # unmasker("Hello I'm a [MASK] model.")
    
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
    
    
    

def main()-> nn.Module:
    
    # 加载数据集
    X_train_combine, Y_train, Y_train_label, X_test_combine, Y_test, Y_test_label, le = build_dataset()
    
    
    # 初始化Tokenizer和模型
    # 加载了预训练的BERT模型和分词器。num_labels=6指定了模型的输出类别数。
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('D:/huggingFace/bert_model')
    
    model = BertForSequenceClassification.from_pretrained('D:/huggingFace/bert_model', num_labels=6)    

    

    # train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    train_dataset = NewsDataset(X_train_combine, Y_train_label, tokenizer)
    test_dataset = NewsDataset(X_test_combine, Y_test_label, tokenizer)
    # shuffle means randomly pick 100 samples as a batch
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100)
    
    # train_loader['input_ids']

    # 训练模型
    
    # create a "device" object
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型的所有参数和缓冲区移动到指定的设备上
    model = model.to(device)
    
    
    
    # why use AdamW:
    # 1. AdamW is a variation of Adam, it uses L2 regularization to utilize "weight decay" to prevent overfitting
    # 2. Bert has large quantity of parameters and very easy to overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    watch_loss = []
    for epoch in range(3):  # 训练3个周期
        model.train()
        for batch in train_loader:
            # 创建了一个字典推导式， k是键， v是k对应的张量， v.to(device)将张量移动到设备上
            # batch字典可能包含了如input_ids，attention_mask等模型需要的输入数据
            batch = {k: v.to(device) for k, v in batch.items()}
            # **操作符是Python中的解包（unpacking）操作符，
            # 它会将字典batch中的键值对作为关键字参数传递给model函数。
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        watch_loss.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        

    plt.figure(figsize=(10, 8))
    plt.plot([i+1 for i in range(len(watch_loss))],[x for x in watch_loss], lw=2, color = 'red', label = 'Training Loss')
    plt.legend()
    plt.show()
    
    
    # 评估模型
    model.eval()
    predictions = []
    with torch.no_grad(): # 使用torch.no_grad()来禁用梯度计算，从而减少内存消耗并加速计算
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())

    # prediction is a (n x 1) matrix ==> a class-number vector
    accuracy = accuracy_score(Y_test_label, predictions)
    precision = precision_score(Y_test_label, predictions)
    recall = recall_score(Y_test_label, predictions)
    f1 = f1_score(Y_test_label, predictions)
    auc = roc_auc_score(Y_test_label, predictions)
    
    print(f"Test Accuracy: {accuracy}")
    print(f"Test precision: {precision}")
    print(f"Test recall: {recall}")
    print(f"Test f1: {f1}")
    print(f"Test auc: {auc}")
    
    
    
    
    # 如何使用bert 模型进行预测？
    # y= model(x) 即可
    
    
    return model, [accuracy, precision, recall, f1, auc]




def build_dataset()->Union[np.ndarray,torch.FloatTensor, torch.LongTensor]:
       
    '''
    创建训练集和测试集
    
    :Y_train: 包含类标签对应整数的列
    :Y_train_label: 包含类标签的列
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
    
    
    # 对 X_train 和 X_test 每行中的特征进行拼接
    X_train_combine:pd.DataFrame = X_train.apply(lambda x: ','.join(map(str, x)), axis=1)
    
    # 转成矩阵
    # X_train_combine: np.ndarray = X_train_combine.values.reshape(-1,1)
    # print(f'X_train_combine = \n{X_train_combine}')
    
    

    X_test_combine:pd.DataFrame = X_test.apply(lambda x: ','.join(map(str, x)), axis=1)
    

    # X_test_combine: np.ndarray = X_test_combine.values.reshape(-1,1)
    print(f'X_test_combine = \n{X_train_combine}')
    
    
    # 将标签列Activity中的string映射成整数，这样模型才可以学习一个函数来预测这个整数
    # 读取activity列中的所有字符串到一个列表 ... 手动赋予编码0-5
    # 创建一个标签编码器对象
    le = LabelEncoder()
    Y_train: np.ndarray = le.fit_transform(Y_train_label)
    Y_test: np.ndarray = le.fit_transform(Y_test_label)

    

    
    # 将所有数据集全部转为Tensor
    X_train = torch.FloatTensor(X_train.to_numpy())
    Y_train = torch.LongTensor(Y_train)
    # Y_train_label = torch.StringTensor(Y_train_label)
    
    
    X_test = torch.FloatTensor(X_test.to_numpy())
    Y_test = torch.LongTensor(Y_test)
    # Y_test_label = torch.FloatTensor(Y_test_label)
    
    

    
    
    print(f"X_train.shape：{X_train.shape}")
    print(f"Y_train.shape：{Y_train.shape}")
    print(f"X_test.shape：{X_test.shape}")
    print(f"Y_test.shape：{Y_test.shape}")
    
    print(f"X_train_combine：{X_train_combine.shape}")
    print(f"X_test_combine：{X_test_combine.shape}")
    print('--------------------------------')
    print(f"Number of feature = {X_train.shape[1]}")
    
    return  X_train_combine, Y_train_label, Y_train, X_test_combine, Y_test, Y_test_label, le




def print_transformer():
    model, model_metrics_list = main()
    return model, model_metrics_list





if __name__ == '__main__':
    main()
    # test_bert()
    # build_dataset()