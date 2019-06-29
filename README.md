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

### 1.4 The covariance Matrix（协方差矩阵）

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

### 1.5 Eigenvectors（特征向量）

在矩阵论中，我们可以这样去理解特征值和特征向量，一个矩阵由一个变换到另一个矩阵，Aα=λα，其中α称为矩阵A 的一个特征向量，λ称为矩阵A的一个特征值。特征向量确定了矩阵变换的方向，特征值确定了矩阵变换的比例。

在协方差矩阵中，协方差矩阵的特征向量又反应了什么物理意义呢？

协方差矩阵的特征向量代表的意思是方差最大的数据所在的方向。在n维数据空间中，第一特征向量指向的是数据方差最大的方向，第二特征向量是与第一特征向量垂直的

数据方差最大的方向，第三特征向量是与第二特征向量垂直的数据方差最大的方向，以此类推。

通常我们还需要对特征向量进行标准化处理，即求模长，然后将向量中每一个元素除以该模长，即为标准化处理。

我们使用python来计算这样一个二维数据的协方差矩阵以及该协方差矩阵的特征值和特征向量，作为我们学习的一个简单的例子

数据如下：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/7.png)

首先计算x维和y维的平均值，代码如下：
```Python
x=[2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]
y=[2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]
means_X=round(float(sum(x)/(len(x))),4)
means_Y=round(float(sum(y)/(len(y))),4)
print("x 维的平均值为：",means_X)
print("y 维的平均值为：",means_Y)

```
用初始数据中x维和y维的每一个数字减去平均值，得到
然后计算协方差：
```python
update_x=[]
update_y=[]
for i in range(0,len(x)):
    var=x[i]-means_X
    update_x.append(var)
    i=i+1
print(update_x)
for i in range(0,len(x)):
    var=y[i]-means_Y
    update_y.append(var)
    i=i+1
print(update_y)
#将两个数组纵向合并
c=np.vstack((update_x,update_y))
print(c)
#调用numpy包计算协方差
cov_c=np.cov(c)
cov_st=np.array(cov_c)
print(type(cov_st))
```
再计算相应的特征值和特征向量：

```python

A,B=np.linalg.eig(cov_st)
print("该协方差矩阵的特征值为：",A)
print("该协方差矩阵的特征向量为：",B)

```
代码运行结果如下图所示：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/8.png)

### 1.6 Choosing components and forming a feature vector（选择主成分并生成特征向量）

一个协方差矩阵有着不同的特征值与特征向量，事实上最高特征值的对应的特征向量就是这个数据集的主成分。

通常来说，一旦协方差矩阵的特征值和特征向量被计算出来了之后，就是按照特征值的大小从高到低依次排列。特征值的大小确定了主成分的重要性。

*主成分分析的基本原理就是：选择特征值较大的作为主成分，从而进行降维。比如：一开始你的数据集是N维的，在进行了协方差矩阵的特征值计算后，
你得到了N个特征值和与这些特征值相对应的特征向量。然后在主成分分析时，你选取了前P个较大的特征值，如此一来，就将原来N维的数据降维到只有P维。这样就起到了降维的效果了。*

### 1.7 Deriving the new data set（选择特征向量生成新的数据集【这个新的数据集也就是降维后的数据集】）

计算新生成的数据集的公式如下：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/9.PNG)

其中rowFeatureVector是由模式矢量作为列组成的矩阵的转置，因此它的行就是原来的模式矢量，而且对应最大特征值的特征矢量在该矩阵的最上一行。rowdataAdjust是每一维数据减去均值后，所组成矩阵的转置，即数据项目在每一列中，每一行是一维，对我们的样本来说即是，第一行为x维上数据，第二行为y维上的数据

正是由于特征向量是两两正交的，那么我们就可以使用任何的特征向量来将原始数据变换到正交的这些坐标轴上。

我们还以前文提过的简单例子来表示。在前文中，我们已经计算出协方差矩阵的特征值及特征向量，接下来，我们选取一个较大的特征值对应的特征向量将原始数据降到一维。做法是：将较大的特征值对应的特征向量转置然后乘以原始数据集，这样就得到新的降维后的一维数据。

# 二、PCA的人脸识别算法（基于Python实现）

## 一、数据集的说明及相关函数的实现

我们使用的是ORL官方数据集，可以从一下网址下载到[ORL下载链接](http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z)
