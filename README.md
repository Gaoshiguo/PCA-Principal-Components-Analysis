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

现在假设我们有一个三个维度的数据，使用一个协方差矩阵将这三维数据的协方差表示如下：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/6.PNG)

设置小练习如下：

请计算一下两个数据的协方差矩阵：
x:(10,39,19,23,28)
y:(43,13,32,21,20)
以及x:(1，-1,4) y:(2,1,3) z:(1,3,-1,)

请计算这两个数据的协方差矩阵

### 1.3 Eigenvectors（特征向量）

在矩阵论中，我们可以这样去理解特征值和特征向量，一个矩阵由一个变换到另一个矩阵，Aα=λα，其中α称为矩阵A 的一个特征向量，λ称为矩阵A的一个特征值。特征向量确定了矩阵变换的方向，特征值确定了矩阵变换的比例。

在协方差矩阵中，协方差矩阵的特征向量又反应了什么物理意义呢？

协方差矩阵的特征向量代表的意思是方差最大的数据所在的方向。在n维数据空间中，第一特征向量指向的是数据方差最大的方向，第二特征向量是与第一特征向量垂直的

数据方差最大的方向，第三特征向量是与第二特征向量垂直的数据方差最大的方向，以此类推。

通常我们还需要对特征向量进行标准化处理，即求模长，然后将向量中每一个元素除以该模长，即为标准化处理。



