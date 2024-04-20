import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# %matplotlib inline

import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV



class SVMModel:
    def __init__(self):
        self.kernel=None
        self.state = None 
    
    
    
    def set_kernel(self, kernel):
        self.kernel = kernel
    
    def set_state(self, state):
        self.state = state
        
    
    def forward(self, x, y= None):
        pass




def build_dataset_svm():
    train = shuffle()












def print_svm():
    pass







if __name__ == '__main__':
    pass