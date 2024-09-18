# Physical Activity Classification Task
## 实验报告
具体细节可以查看 实验报告.pdf

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
* 我们比较了 accuracy, recall, precision, f1, roc, auc 这几个指标，然后就是交叉验证的平均loss
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



## 实验假设
![image](https://github.com/user-attachments/assets/6f42e543-87b1-4fd5-982a-76ac56ec9309)
![image](https://github.com/user-attachments/assets/eede6e50-b49d-464b-bf6f-577053a3f20f)



## 实验目标
1. 我们的第一个目标是将测试数据集（test.cv）中的每个样本分类为6个类别之一（Laying，Standing，Sitting，Walking，Walking_upstairs，Walking_downstairs），使用5种不同的机器学习模型（神经网络，SVM，随机森林，Bert，A*优化神经网络）来实现这一目标。
2. 为此，我们首先需要训练我们的分类器，训练集存储在“train.csv”文件中，每行包含563列，其中前561列称为样本特征，最后一列“Activity”（第563列）是类别标签，它存储了一个样本所属的类别名称（例如Standing）。

3. 我们的第二个目标是使用matplotlib获取不同模型的性能图（准确率、训练损失、交叉验证损失、精确率、召回率、f1得分、auc），以折线图、条形图、ROC曲线、混淆矩阵和分类报告表的形式展现。

4. 我们的第三个目标是比较这5个模型之间的所有性能指标，并分析指标背后的原因。

5. 我们的第四个目标是查看Bert模型是否可以应用于多浮点数特征样本的分类任务【其中涉及到将多浮点数样本转为单一的文本样本】，并获得类似的性能。

6. 我们的最终目标是设计和实现一个A*算法，能够使用元启发搜索从各种不同的【隐藏层数，隐单元数，激活函数种类（sigmoid, relu, tanh）】组合中，找到一个最佳组合，使得应用这个组合下的神经网络的分类性能可以达到并超越SVM在相同任务上的最佳性能。

## 实验结果
![image](https://github.com/user-attachments/assets/3a3ace0a-4135-472d-898f-18c63a9d8a07)

- 我们选出了两种性能最好的模型，分别是GridSearch优化后的SVM，以及A*优化后的神经网络（自定义TorchModel）。
- 可以看出，A*优化后的NN比没有优化过的NN，在所有指标上，都高出了%2~%4。
- 但是即使有A*的加持，SVM依然比优化后的NN高出2%左右。

![image](https://github.com/user-attachments/assets/ed66cb4e-8670-48e6-ad63-785b6b7c0bb9)



## 我们自定义的神经网络结构（TorchModel）
![image](https://github.com/user-attachments/assets/404e3515-de8c-4d3a-a7f9-f0d98111caf0)
![image](https://github.com/user-attachments/assets/b3c8ec7f-73fa-417b-bde9-83273a6d9b06)


## 在SVM中使用GridSearch寻找最佳核函数
![image](https://github.com/user-attachments/assets/cb7eb23a-d613-46e5-814f-dd9cf12a079e)
![image](https://github.com/user-attachments/assets/74c95890-8a98-4543-8f9a-bebdddb303f1)


## 使用A*搜索算法搜索最优的神经网络超参数组合【hidden_size, 】

