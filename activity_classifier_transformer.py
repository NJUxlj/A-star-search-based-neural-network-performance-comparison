import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# BertTokenizer: divide a text into many tokens
# BertForSequenceClassification: classify a text into certain class
# AdamW: a optimizer
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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
texts = ["政府今日宣布了新的经济政策", "球队在昨晚的比赛中表现出色", "股市今日大幅上涨"]
labels = [1, 2, 1]  # 假设1代表经济，2代表体育

'''
NewsDataset: 自定义的数据集类，继承自PyTorch的Dataset类。它接受文本、标签和分词器作为输入，并处理文本的编码。
encode_plus: BERT分词器的方法，用于将文本转换为模型可以理解的格式，包括添加特殊令牌、截断、填充等。
'''

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }





def main():
    # 初始化Tokenizer和模型
    # 加载了预训练的BERT模型和分词器。num_labels=3指定了模型的输出类别数。
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)    


    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):  # 训练3个周期
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    
    # 评估模型
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




def build_dataset():
       
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
    
    
    # 对 X_train 和 X_test 每行中的特诊进行拼接
    X_train_combine = X_train.apply(lambda x: ','.join(map(str, x)), axis=1)
    
    # 转成矩阵
    X_train_combine: np.ndarray = X_train_combine.values.reshape(-1,1)
    # print(f'X_train_combine = \n{X_train_combine}')
    
    

    X_test_combine = X_test.apply(lambda x: ','.join(map(str, x)), axis=1)
    

    X_test_combine: np.ndarray = X_test_combine.values.reshape(-1,1)
    print(f'X_test_combine = \n{X_train_combine}')
    
    
    # 将标签列Activity中的string映射成整数，这样模型才可以学习一个函数来预测这个整数
    # 读取activity列中的所有字符串到一个列表 ... 手动赋予编码0-5
    # 创建一个标签编码器对象
    le = LabelEncoder()
    Y_train: np.ndarray = le.fit_transform(Y_train_label)
    Y_test: np.ndarray = le.fit_transform(Y_test_label)

    

    
    # 将所有数据集全部转为Tensor
    # X_train = torch.FloatTensor(X_train.to_numpy())
    # Y_train = torch.FloatTensor(Y_train)
    # X_test = torch.FloatTensor(X_test.to_numpy())
    # Y_test = torch.FloatTensor(Y_test)
    

    
    
    print(f"X_train.shape：{X_train.shape}")
    print(f"Y_train.shape：{Y_train.shape}")
    print(f"X_test.shape：{X_test.shape}")
    print(f"Y_test.shape：{Y_test.shape}")
    
    print(f"X_train_combine：{X_train_combine.shape}")
    print(f"X_test_combine：{X_test_combine.shape}")
    print('--------------------------------')
    print(f"Number of feature = {X_train.shape[1]}")
    
    return  X_train_combine, Y_train, X_test_combine, Y_test, Y_test_label, le




def print_transformer():
    main()






if __name__ == '__main__':
    # main()
    build_dataset()