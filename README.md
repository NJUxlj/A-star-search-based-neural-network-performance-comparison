# cpt406-group1
CPT406 CourseWork Activity Classification

## 项目环境配置
* 打开anaconda 控制台
* 输入：conda activate 环境名
* 输入以下命令
```python
pip install torch==1.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install scikit-learn
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```


## 项目文件组织
* 总体上看， 包含两个分类模型（classifier），第一个classifier使用神经网络实现的，用到的包是torch；
    第二个classifier是用SVM实现的，用到的包是sklearn.
* activity_classification_sklearn.py 中存放的是SVM
* activity_classification_torch.py 中存放的是Neural Network


## 项目进展
* 现在代码已经可以运行， 你们可以参照这个先把report写掉70%。剩余的30%, 我将测试多种SVM核函数并更新到代码里，预计明晚之前弄好。
