import torch
from torch.utils.data import DataLoader, Dataset

# BertTokenizer: divide a text into many tokens
# BertForSequenceClassification: classify a text into certain class
# AdamW: a optimizer
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np





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

















def print_transformer():
    pass






if __name__ == '__main__':
    pass