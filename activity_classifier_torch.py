import torch.nn
import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class TorchModule(nn.Module):
    def __init__(self, inputsize: int):
        ''''
        inputsize: 输入样本的特征数量
        '''
        super(TorchModel,self).__init__() # 调用父类构造器
        self.linear = torch.F.linear(input, weight)
        
        
        pass
    
    
    






def build_sample():
    X= np.random.random()
    y = 




# 执行训练任务
def main():
    pass