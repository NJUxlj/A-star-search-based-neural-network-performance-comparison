import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# %matplotlib inline

import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV



class SVMModel:
    def __init__(self):
        self.kernel=None
        self.state = None 
    
    
    
    def set_kernel(self, kernel):
        '''
        设置核函数
        '''
        self.kernel = kernel
    
    def set_state(self, state):
        self.state = state
        
    def set_kernel_params(self):
        '''
        设置核函数的参数，比如：
        1. 多项式核的 h和c
        2. rbf和的sigma， 8它控制着核函数的宽度， 影响模型的灵敏度
        '''
        pass
    
    def forward(self, x, y= None):
        pass




def build_dataset_svm():
    train = shuffle()












def print_svm():
    pass







if __name__ == '__main__':
    pass