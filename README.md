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

下载后的数据集是这个样子的：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/10.png)

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/11.png)

该数据集表示的是一共有40个人的人脸图像，其中每一个人有10张人脸图像。相应的PGM文件为说明。

我们需要用到的第三方包有numpy主要用于科学计算，cv主要用于图像处理，matplotlib主要用于训练结果展示

首先定义一个函数用于将人脸图像矢量化为一个向量，向量的大小与图片的像素有关，代码如下：
```python
# 图片矢量化
def img2vector(image):
    img = cv2.imread(image, 0)  # 读取图片
    rows, cols = img.shape  #获取图片的像素
    imgVector = np.zeros((1, rows * cols)) #初始值均设置为0，大小就是图片像素的大小
    imgVector = np.reshape(img, (1, rows * cols))#使用imgVector变量作为一个向量存储图片矢量化信息
    return imgVector
 ```
 接下来定义一个函数用来选取训练图片，并对每张图片进行前面定义过的矢量化处理
 
```python
def load_orl(k):#参数K代表选择K张图片作为训练图片使用
    '''
    对训练数据集进行数组初始化，用0填充，每张图片尺寸都定为112*92,
    现在共有40个人，每个人都选择k张，则整个训练集大小为40*k,112*92
    '''
    train_face = np.zeros((40 * k, 112 * 92))
    train_label = np.zeros(40 * k)  # [0,0,.....0](共40*k个0)
    test_face = np.zeros((40 * (10 - k), 112 * 92))
    test_label = np.zeros(40 * (10 - k))
    # sample=random.sample(range(10),k)#每个人都有的10张照片中，随机选取k张作为训练样本(10个里面随机选取K个成为一个列表)
    sample = random.permutation(10) + 1  # 随机排序1-10 (0-9）+1
    for i in range(40):  # 共有40个人
        people_num = i + 1
        for j in range(10):  # 每个人都有10张照片
            image = orlpath + '/s' + str(people_num) + '/' + str(sample[j]) + '.jpg'
            # 读取图片并进行矢量化
            img = img2vector(image)
            if j < k:
                # 构成训练集
                train_face[i * k + j, :] = img
                train_label[i * k + j] = people_num
            else:
                # 构成测试集
                test_face[i * (10 - k) + (j - k), :] = img
                test_label[i * (10 - k) + (j - k)] = people_num

    return train_face, train_label, test_face, test_label
   ```
前期将所有训练图片矢量化之后，开始进行PCA算法的降维操作
```python
def PCA(data, r):#参数r代表降低到r维
    data = np.float32(np.mat(data))
    rows, cols = np.shape(data)
    data_mean = np.mean(data, 0)  # 对列求平均值
    A = data - np.tile(data_mean, (rows, 1))  # 将所有样例减去对应均值得到A
    C = A * A.T  # 得到协方差矩阵
    D, V = np.linalg.eig(C)  # 求协方差矩阵的特征值和特征向量
    V_r = V[:, 0:r]  # 按列取前r个特征向量
    V_r = A.T * V_r  # 小矩阵特征向量向大矩阵特征向量过渡
    for i in range(r):
        V_r[:, i] = V_r[:, i] / np.linalg.norm(V_r[:, i])  # 特征向量归一化

    final_data = A * V_r
    return final_data, data_mean, V_r
  ```
  
  最后我们进行初次训练，随机选取每个人物的五张图片作为训练图片使用。将降低的维数设定为10维，查看一下训练效果如何。
  ```python
  def face_recongize():
    #对每一个人随机选取5张照片作为训练数据
    train_face, train_label, test_face, test_label = load_orl(5)#随机选择每个人物的5张图片作为训练数据

    x_value = []
    y_value = []
    #将图片降维到10维
    data_train_new, data_mean, V_r = PCA(train_face, 10)
    num_train = data_train_new.shape[0]  # 训练脸总数
    num_test = test_face.shape[0]  # 测试脸总数
    temp_face = test_face - np.tile(data_mean, (num_test, 1))
    data_test_new = temp_face * V_r  # 得到测试脸在特征向量下的数据
    data_test_new = np.array(data_test_new)  # mat change to array
    data_train_new = np.array(data_train_new)

    true_num = 0
    for i in range(num_test):
        testFace = data_test_new[i, :]
        diffMat = data_train_new - np.tile(testFace, (num_train, 1))  # 训练数据与测试脸之间距离
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)  # 按行求和
        sortedDistIndicies = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
        indexMin = sortedDistIndicies[0]  # 距离最近的索引
        if train_label[indexMin] == test_label[i]:
            true_num += 1
        else:
            pass

    accuracy = float(true_num) / num_test
    x_value.append(5)
    y_value.append(round(accuracy, 2))

    print('当对每个人随机选择%d张照片降低至%d维进行训练时，The classify accuracy is: %.2f%%' % (5,10, accuracy * 100))

```
最终训练得到的结果如下：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/12.png)

为了对比实验，我们分别选取5张、7张、9张，还是降低到10维进行对比实验：

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/13.png)

可以看出来随着训练集的图片选取的不断增多，训练准确率在不断增加。这是因为训练的样本多了，但是我们如果选择全部的10张图片作为训练样本的话就会导致训练结果过拟合。

再一次进行对比实验，我们在选用同样是7张图片作为训练样本的情况下，将降低的维数改为10维、20维、30维，查看训练效果如何。

![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/14.png)
![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/15.png)
![image](https://github.com/Gaoshiguo/PCA-Principal-Components-Analysis/blob/master/image/16.png)

## 经过多次试验总结发现，训练样本越多训练效果越好，训练维数越高效果越好，但并不是绝对的，本次试验就发现，在选取的训练样本相同的情况下，降低至40维的效果反而不如降低至30维的效果
