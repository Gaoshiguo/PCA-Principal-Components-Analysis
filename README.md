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

### 1.1 Variance（标准差）

Variance is another measure of the spread of data in a data set. In fact it is almost identical to the standard deviation.（方差是数据集中数据分布的另一种度量。实际上，它几乎与标准差相同）

标准差的数学表达公式如下：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/3.PNG)
