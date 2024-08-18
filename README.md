# Physical Activity Classification Task


## 项目环境配置
* 下载anaconda 并配置环境变量，不会的自己去搜
* 创建一个新的conda环境
* conda create --name 环境名字 python=3.8
* conda env list，查看当前有哪些环境
* conda activate 你的环境名

* 打开anaconda 控制台  (Anaconda Prompt)
* 输入：conda activate 环境名
* 输入以下命令
```python
pip install torch==1.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/

如果1.10.0版本的torch下载不了，那就直接：pip install torch

pip install scikit-learn
pip install transformers
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```


## 项目文件组织
* 总体上看， 包含两个分类模型（classifier）
    * 第一个classifier使用神经网络实现的，用到的包是torch.
    * 第二个classifier是用SVM实现的，用到的包是sklearn.
* activity_classification_sklearn.py 中存放的是SVM
* activity_classification_torch.py 中存放的是Neural Network
* activity_classification_transformer.py 中存放的是基于transformer架构的Bert模型，用于text分类
* activity_classification_decisionTree.py 中存放的是决策树模型
* 所有的模型会合并到一个runner.py文件中。



## 项目进展
* 模型已经完工。
* 大家也可以手动比较， 就是比价 accuracy, recall, precision, f1, roc, auc 这几个指标，然后就是交叉验证的平均loss
* 
* main分支几乎不会更改了，但会修复bug
* ### 最新进展请移步 dev分支 查看



## IDE环境搭建
* 下载vscode
* 双击打开任意一个.py文件
* 打开应用商店， 下载如下拓展
    * ![image](https://github.com/NJUxlj/cpt406-group1/assets/86636180/b6c0094b-3c55-4c0c-960e-e97d86dd52a8)
* 点击左侧垂直任务栏上的python图标，在如图所示配置你的conda运行环境，并点击如图的五角星标志设置当前环境为你想要的那一个
    * ![image](https://github.com/NJUxlj/cpt406-group1/assets/86636180/73ba2952-0fa0-4417-9508-fee5b07ce8fa)
    * ![image](https://github.com/NJUxlj/cpt406-group1/assets/86636180/7a056a71-1d31-4f67-967c-45c232e7f55e)
* 点击编辑器右上角的一个run code箭头， 或者用快捷键 ctrl+alt+N 即可运行。你应该能在控制台看到打印结果
* 如果你不想用vscode，打开jupyter notebook， 把代码和csv文件复制进去运行即可。

     

## 如何运行项目
* 把项目代码打包解压到本地后， 打开runner.py文件
* 运行即可
* 如果你只想看单个模型的表现的话，注释掉其他模型对应的代码即可
