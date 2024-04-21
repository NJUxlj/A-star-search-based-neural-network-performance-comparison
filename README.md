# cpt406-group1
CPT406 CourseWork Activity Classification

## 注：你们自己不要动代码

## 项目环境配置
* 打开anaconda 控制台
* 输入：conda activate 环境名
* 输入以下命令
```python
pip install torch==1.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
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
* 后期这两个文件我会合并到一个runner.py文件中。
* 大家只要写第1，2 个模型就行了， 第3,4个看后期进度


## 项目进展
* 模型已经完工， 针对两个模型的性能比较，我还需要加一点可视化的代码
* 大家也可以手动比较， 就是比价 accuracy, recall, precision, f1, roc, auc 这几个指标，然后就是交叉验证的平均loss



## IDE环境搭建
* 前提：如果你会git的命令，那就直接把仓库拉到本地，因为我经常会更新，这样更方便。如果你不会，拿就直接把项目.zip下载到本地解压。
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