# PCA-Principal-Components-Analysis
# PCA是一项常用的在高维数据中寻找特征的降维技术，目前主要用于图片识别和图片压缩领域中。本文主要讲两个部分：

# 一、PCA的算法原理。

# 二、PCA的人脸识别算法

## 一、PCA的算法原理

首先需要知道几个相关的数学概念，这是我们进行PCA分析的基础

标准差（Standard Deviation）、方差（Variance）、协方差（Covariance）、特征向量（eigenvectors）、特征值（eigenvalues）


### 1.1 Standard Deviation（标准差）
标准差就是用来描述一组数据与平均值得偏离程度，反映了一组数据的波动情况，平均值数学表达公式如下：
![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/1.PNG)

标准差的表达公式如下：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/2.PNG)

需要注意的是分母是选择`n`还是`n-1`，取决于你选取的数据是整个完整数据还是数据中的一部分

### 1.2 Variance（方差）

Variance is another measure of the spread of data in a data set. In fact it is almost identical to the standard deviation.（方差是数据集中数据分布的另一种度量。实际上，它几乎与标准差相同）

方差的数学表达公式如下：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/3.PNG)

### 1.3 Covariance（协方差）
标准差与方差只针对一维数据进行衡量的指标，协方差是针对二维数据或者是更高维的数据进行的衡量指标，主要用来表示多维度与平均值的偏离程度。

协方差的数学表达公式如下：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/4.PNG)

### 1.2 The covariance Matrix（协方差矩阵）

协方差矩阵主要是用于当数据的维度超过3或者更多的时候，我们可以通过一个矩阵来存储各个维度的协方差，这个矩阵就被称为“协方差矩阵”。

用数学方法来表示一个N为数据的协方差矩阵可以表示为：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/5.PNG)





